"""
Real-time audio capture with VAD and noise reduction
"""

import numpy as np
import threading
import queue
import time
from typing import Optional, Callable, Tuple, List, Dict
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("⚠️  sounddevice not available. Install with: pip install sounddevice")

from .config import AudioConfig


@dataclass
class AudioChunk:
    """Audio chunk with metadata"""
    data: np.ndarray
    timestamp: float
    duration: float
    is_speech: bool = False
    energy: float = 0.0


class AudioCapture:
    """
    Real-time audio capture from microphone
    
    Features:
    - Continuous audio capture
    - Voice Activity Detection (VAD)
    - Noise gate
    - Buffer management
    - Callback-based processing
    """
    
    def __init__(self, config: AudioConfig):
        """
        Initialize audio capture
        
        Args:
            config: Audio configuration
        """
        self.config = config
        self.is_running = False
        self.is_paused = False
        
        # Audio buffers
        self.audio_queue = queue.Queue()
        self.speech_buffer = []
        self.silence_buffer = []
        
        # VAD state
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None
        self.silence_duration = 0.0
        
        # Statistics
        self.stats = {
            'total_chunks': 0,
            'speech_chunks': 0,
            'silence_chunks': 0,
            'total_duration': 0.0,
            'speech_duration': 0.0
        }
        
        # Callbacks
        self.on_speech_start: Optional[Callable] = None
        self.on_speech_end: Optional[Callable] = None
        self.on_chunk: Optional[Callable] = None
        
        # Thread management
        self.capture_thread: Optional[threading.Thread] = None
        self.processing_thread: Optional[threading.Thread] = None
        
        # Audio stream
        self.stream = None
        
        # Buffer for processing
        self.buffer = []
        self.buffer_duration = 0.0
    
    def start(self):
        """Start audio capture"""
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError("sounddevice is required for audio capture")
        
        if self.is_running:
            return
        
        print("🎤 Starting audio capture...")
        
        self.is_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        
        # Start audio stream
        self.stream = sd.InputStream(
            channels=self.config.channels,
            samplerate=self.config.sample_rate,
            blocksize=self.config.chunk_size,
            callback=self._audio_callback
        )
        self.stream.start()
        
        print("✅ Audio capture started")
    
    def stop(self):
        """Stop audio capture"""
        if not self.is_running:
            return
        
        print("⏹️  Stopping audio capture...")
        
        self.is_running = False
        
        # Stop stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
        
        # Stop processing thread
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        # Process any remaining buffer
        if self.speech_buffer:
            self._process_speech_end()
        
        print("✅ Audio capture stopped")
    
    def pause(self):
        """Pause audio capture"""
        self.is_paused = True
        print("⏸️  Audio capture paused")
    
    def resume(self):
        """Resume audio capture"""
        self.is_paused = False
        print("▶️  Audio capture resumed")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """
        Callback for audio stream
        
        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Time information
            status: Status flags
        """
        if status:
            print(f"⚠️  Audio status: {status}")
        
        if self.is_paused:
            return
        
        # Extract audio data
        audio_data = indata[:, 0].copy()
        
        # Add to queue
        self.audio_queue.put(audio_data)
    
    def _processing_loop(self):
        """Main processing loop for audio chunks"""
        while self.is_running:
            try:
                # Get audio chunk
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # Process chunk
                self._process_chunk(audio_data)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error in processing loop: {e}")
    
    def _process_chunk(self, audio_data: np.ndarray):
        """
        Process a single audio chunk
        
        Args:
            audio_data: Audio data (1D numpy array)
        """
        # Calculate energy
        energy = self._calculate_energy(audio_data)
        
        # Apply noise gate
        if energy < self.config.noise_gate_threshold:
            audio_data = audio_data * 0.1  # Reduce noise
        
        # VAD detection
        is_speech = self._detect_speech(audio_data, energy)
        
        # Create chunk
        chunk = AudioChunk(
            data=audio_data,
            timestamp=time.time(),
            duration=len(audio_data) / self.config.sample_rate,
            is_speech=is_speech,
            energy=energy
        )
        
        # Update statistics
        self.stats['total_chunks'] += 1
        self.stats['total_duration'] += chunk.duration
        
        if is_speech:
            self.stats['speech_chunks'] += 1
            self.stats['speech_duration'] += chunk.duration
        else:
            self.stats['silence_chunks'] += 1
        
        # Handle speech/silence transitions
        if is_speech:
            self._handle_speech(chunk)
        else:
            self._handle_silence(chunk)
        
        # Trigger chunk callback
        if self.on_chunk:
            self.on_chunk(chunk)
    
    def _calculate_energy(self, audio_data: np.ndarray) -> float:
        """
        Calculate audio energy
        
        Args:
            audio_data: Audio data
            
        Returns:
            Energy value (RMS)
        """
        if len(audio_data) == 0:
            return 0.0
        
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_data ** 2))
        return float(rms)
    
    def _detect_speech(self, audio_data: np.ndarray, energy: float) -> bool:
        """
        Detect if audio contains speech using simple VAD
        
        Args:
            audio_data: Audio data
            energy: Audio energy
            
        Returns:
            True if speech is detected
        """
        # Simple energy-based VAD
        if energy > self.config.vad_threshold:
            return True
        
        # Check for speech pattern (zero-crossing rate)
        if len(audio_data) > 1:
            zero_crossings = np.sum(np.diff(np.sign(audio_data)) != 0)
            zcr = zero_crossings / len(audio_data)
            
            # Speech typically has moderate ZCR
            if 0.05 < zcr < 0.5 and energy > self.config.vad_threshold * 0.5:
                return True
        
        return False
    
    def _handle_speech(self, chunk: AudioChunk):
        """
        Handle speech detection
        
        Args:
            chunk: Audio chunk with speech
        """
        current_time = time.time()
        
        # If we weren't speaking, this is speech start
        if not self.is_speaking:
            self.is_speaking = True
            self.speech_start_time = current_time
            
            # Trigger speech start callback
            if self.on_speech_start:
                self.on_speech_start()
            
            print("🎤 Speech started")
        
        # Add to speech buffer
        self.speech_buffer.append(chunk.data)
        
        # Update last speech time
        self.last_speech_time = current_time
        self.silence_duration = 0.0
        
        # Check if buffer is full
        buffer_duration = sum(len(c) for c in self.speech_buffer) / self.config.sample_rate
        
        if buffer_duration >= self.config.max_buffer_duration:
            self._process_speech_end()
    
    def _handle_silence(self, chunk: AudioChunk):
        """
        Handle silence detection
        
        Args:
            chunk: Audio chunk with silence
        """
        current_time = time.time()
        
        # If we were speaking, accumulate silence
        if self.is_speaking:
            self.silence_duration += chunk.duration
            
            # Check if silence is long enough to end speech
            if self.silence_duration >= self.config.vad_min_silence:
                self._process_speech_end()
    
    def _process_speech_end(self):
        """Process end of speech segment"""
        if not self.speech_buffer:
            return
        
        # Combine speech buffer
        speech_data = np.concatenate(self.speech_buffer)
        speech_duration = len(speech_data) / self.config.sample_rate
        
        # Check if speech is long enough
        if speech_duration >= self.config.vad_min_speech:
            # Trigger speech end callback
            if self.on_speech_end:
                self.on_speech_end(speech_data, speech_duration)
            
            print(f"🎤 Speech ended: {speech_duration:.2f}s")
        
        # Clear buffer
        self.speech_buffer = []
        self.is_speaking = False
        self.silence_duration = 0.0
    
    def get_current_buffer(self) -> Optional[np.ndarray]:
        """
        Get current audio buffer for streaming processing
        
        Returns:
            Current audio buffer or None if empty
        """
        if not self.speech_buffer:
            return None
        
        # Combine all chunks
        return np.concatenate(self.speech_buffer)
    
    def clear_buffer(self):
        """Clear audio buffers"""
        self.speech_buffer = []
        self.silence_buffer = []
    
    def get_statistics(self) -> Dict:
        """Get capture statistics"""
        return self.stats.copy()
    
    def list_devices(self) -> List[Dict]:
        """List available audio devices"""
        if not SOUNDDEVICE_AVAILABLE:
            return []
        
        devices = sd.query_devices()
        input_devices = []
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': device['default_samplerate']
                })
        
        return input_devices
