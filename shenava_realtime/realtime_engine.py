"""
Real-time ASR engine with streaming capabilities
"""

import time
import threading
import queue
import numpy as np
import torch
from typing import Optional, Callable, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

from .config import ASRConfig, AppConfig
from .audio_capture import AudioCapture, AudioChunk
from .postprocessor import PostProcessor
from .utils import TextBuffer, ConfidenceCalculator


@dataclass
class TranscriptionState:
    """Current transcription state"""
    is_processing: bool = False
    current_text: str = ""
    partial_text: str = ""
    last_update: float = 0.0
    confidence: float = 0.0
    words_processed: int = 0
    chunks_processed: int = 0


class RealtimeASR:
    """
    Real-time ASR engine with streaming transcription
    
    Features:
    - Streaming audio processing
    - Partial and final results
    - Confidence scoring
    - Text accumulation
    - Callback-based output
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize real-time ASR engine
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.asr_config = config.asr
        
        print("🚀 Initializing Real-time ASR Engine...")
        print(f"   Model: {self.asr_config.model_name}")
        print(f"   Device: {self.asr_config.device}")
        
        # Load model
        self._load_model()
        
        # Initialize components
        self.postprocessor = PostProcessor(self.asr_config)
        self.text_buffer = TextBuffer()
        self.confidence_calc = ConfidenceCalculator()
        
        # State
        self.state = TranscriptionState()
        self.is_running = False
        
        # Audio capture
        self.audio_capture = AudioCapture(self.config.audio)
        self._setup_audio_callbacks()
        
        # Partial-result throttle timestamp
        self._last_partial_time = 0.0
        
        # Processing queue
        self.processing_queue = queue.Queue()
        self.processing_thread: Optional[threading.Thread] = None
        
        # Output callbacks
        self.on_partial_result: Optional[Callable[[str, float], None]] = None
        self.on_final_result: Optional[Callable[[str, float], None]] = None
        self.on_confidence_update: Optional[Callable[[float], None]] = None
        
        # Statistics
        self.stats = {
            'total_processing_time': 0.0,
            'total_audio_duration': 0.0,
            'num_chunks_processed': 0,
            'num_final_results': 0,
            'average_confidence': 0.0
        }
        
        print("✅ Real-time ASR Engine initialized")
    
    def _load_model(self):
        """Load ASR model"""
        try:
            import nemo.collections.asr as nemo_asr
            
            # Prefer local .nemo file when configured (offline, no download)
            model_path = getattr(self.asr_config, 'model_path', None)
            if model_path and Path(model_path).exists():
                print(f"   Loading local model: {model_path}")
                self.model = nemo_asr.models.ASRModel.restore_from(model_path)
            else:
                if model_path:
                    print(f"   ⚠️  Local model not found at {model_path}; falling back to HuggingFace download")
                # Load model
                self.model = nemo_asr.models.ASRModel.from_pretrained(
                    self.asr_config.model_name
                )
            
            # Move to device
            device = torch.device(self.asr_config.device if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)
            self.device = device
            
            # Set to evaluation mode
            self.model.eval()
            
            # CPU inference tuning
            torch.set_num_threads(getattr(self.asr_config, 'num_threads', 4))
            
            # Hybrid RNNT/CTC models: CTC decode is ~16x faster on CPU and is the
            # decode path used by this model's published deployment exports
            decoder_type = getattr(self.asr_config, 'decoder_type', None)
            if decoder_type and hasattr(self.model, 'change_decoding_strategy'):
                try:
                    self.model.change_decoding_strategy(decoder_type=decoder_type)
                    print(f"   Decoder: {decoder_type}")
                except Exception as e:
                    print(f"   ⚠️  Could not switch decoder to '{decoder_type}': {e}")
            
            # Set context window
            self._set_context_window()
            
            print(f"✅ Model loaded on {device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def _set_context_window(self):
        """Set model context window for streaming"""
        context = self.asr_config.context_size
        
        try:
            if hasattr(self.model, 'change_context'):
                self.model.change_context(context)
        except:
            pass
    
    def _setup_audio_callbacks(self):
        """Setup audio capture callbacks"""
        self.audio_capture.on_speech_start = self._on_speech_start
        self.audio_capture.on_speech_end = self._on_speech_end
        self.audio_capture.on_chunk = self._on_audio_chunk
    
    def start(self):
        """Start real-time transcription"""
        if self.is_running:
            return
        
        print("\n" + "="*60)
        print("🎙️  STARTING REAL-TIME TRANSCRIPTION")
        print("="*60)
        print(f"   Mode: {self.config.output_mode.value}")
        print(f"   Overlay: {'Enabled' if self.config.overlay.enabled else 'Disabled'}")
        print(f"   Injector: {'Enabled' if self.config.injector.enabled else 'Disabled'}")
        print("="*60)
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, name="asr-processing", daemon=True)
        self.processing_thread.start()
        
        try:
            self.audio_capture.start()
        except Exception:
            self.is_running = False
            self.processing_thread.join(timeout=2.0)
            self.processing_thread = None
            raise
        
        print("\n🎤 Listening... (Press Ctrl+Alt+R to toggle)")
        print("   Ctrl+Alt+O: Toggle overlay")
        print("   Ctrl+Alt+I: Toggle injector")
        print("   Ctrl+Alt+Q: Quit\n")
    
    def stop(self):
        """Stop real-time transcription"""
        if not self.is_running:
            return
        
        print("\n⏹️  Stopping real-time transcription...")
        
        self.is_running = False
        
        self.audio_capture.stop()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
            self.processing_thread = None
        
        self.state.is_processing = False
        self.state.partial_text = ""
        # Process any remaining final results
        self._process_remaining()
        
        # Save transcript
        if self.config.save_transcripts:
            self._save_transcript()
        
        print("✅ Real-time transcription stopped")
        print(f"   Total words: {self.state.words_processed}")
        print(f"   Total chunks: {self.state.chunks_processed}")
        print(f"   Average confidence: {self.stats['average_confidence']:.3f}")
    
    def _on_speech_start(self):
        """Callback for speech start"""
        self.state.is_processing = True
        if self.config.debug:
            print("🎤 Speech detected")
    
    def _on_speech_end(self, audio_data: np.ndarray, duration: float):
        """
        Callback for speech end
        
        Args:
            audio_data: Speech audio data
            duration: Speech duration in seconds
        """
        # Add to processing queue
        self.processing_queue.put({
            'type': 'final',
            'audio': audio_data,
            'duration': duration,
            'timestamp': time.time()
        })
    
    def _on_audio_chunk(self, chunk: AudioChunk):
        """
        Callback for audio chunk
        
        Args:
            chunk: Audio chunk
        """
        # Add to processing queue for partial results, throttled to avoid
        # flooding inference with a call per 64ms chunk
        min_interval = getattr(self.config.audio, 'partial_buffer_duration', 0.5)
        if (
            chunk.is_speech
            and self.state.is_processing
            and chunk.timestamp - self._last_partial_time >= min_interval
        ):
            self._last_partial_time = chunk.timestamp
            self.processing_queue.put({
                'type': 'partial',
                'audio': self.audio_capture.get_current_buffer(),
                'duration': chunk.duration,
                'timestamp': chunk.timestamp
            })
    
    def _processing_loop(self):
        """Main processing loop"""
        while self.is_running:
            try:
                # Get item from queue
                item = self.processing_queue.get(timeout=0.1)
                
                # Process based on type
                if item['type'] == 'partial':
                    self._process_partial(item['audio'])
                elif item['type'] == 'final':
                    self._process_final(item['audio'], item['duration'])
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error in processing loop: {e}")
    
    def _infer(self, audio_data: np.ndarray):
        """
        Single shared inference path for all transcriptions
        
        Args:
            audio_data: 1D float32 audio
            
        Returns:
            (text, confidence)
        """
        audio = np.asarray(audio_data, dtype=np.float32)
        with torch.no_grad():
            outputs = self.model.transcribe(
                [audio],
                batch_size=1,
                verbose=False,
                return_hypotheses=True
            )
        return self._extract_result(outputs)
    
    def _process_partial(self, audio_data: np.ndarray):
        """
        Process partial audio for streaming results
        
        Args:
            audio_data: Audio data
        """
        try:
            text, confidence = self._infer(audio_data)
            
            if text:
                # Update state
                self.state.partial_text = text
                self.state.last_update = time.time()
                
                # Trigger partial result callback
                if self.on_partial_result:
                    self.on_partial_result(text, confidence)
                
                if self.config.debug:
                    print(f"   Partial: {text}")
            
        except Exception as e:
            if self.config.debug:
                print(f"⚠️  Partial processing error: {e}")
    
    def _process_final(self, audio_data: np.ndarray, duration: float):
        """
        Process final audio segment
        
        Args:
            audio_data: Audio data
            duration: Audio duration
        """
        try:
            start_time = time.time()
            
            text, confidence = self._infer(audio_data)
            
            self.state.is_processing = False
            if text:
                # Post-process text
                processed_text = self.postprocessor.process(text)
                if not processed_text:
                    return
                
                # Update state
                self.state.current_text = processed_text
                self.state.partial_text = ""
                self.state.confidence = confidence
                self.state.words_processed += len(processed_text.split())
                self.state.chunks_processed += 1
                self.state.last_update = time.time()
                self.state.is_processing = False
                
                # Add to text buffer
                self.text_buffer.add_text(processed_text)
                
                # Update statistics
                processing_time = time.time() - start_time
                self.stats['total_processing_time'] += processing_time
                self.stats['total_audio_duration'] += duration
                self.stats['num_chunks_processed'] += 1
                self.stats['num_final_results'] += 1
                self.stats['average_confidence'] = (
                    (self.stats['average_confidence'] * (self.stats['num_final_results'] - 1) + confidence)
                    / self.stats['num_final_results']
                )
                
                # Trigger final result callback
                if self.on_final_result:
                    self.on_final_result(processed_text, confidence)
                
                print(f"   ✓ [{duration:.1f}s] {processed_text}")
            
        except Exception as e:
            print(f"❌ Final processing error: {e}")
    
    def _extract_result(self, outputs: Any) -> Tuple[str, float]:
        """
        Extract text and confidence from model outputs
        
        Args:
            outputs: Model outputs
            
        Returns:
            Tuple of (text, confidence)
        """
        text = ""
        confidence = 0.0
        
        try:
            # NeMo normally returns a list, but some versions return a single
            # hypothesis/string. Accept both forms.
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            if len(outputs) > 0:
                output = outputs[0]
                
                # NeMo Hypothesis objects (return_hypotheses=True)
                if hasattr(output, 'text') and not isinstance(output, str):
                    text = output.text or ''
                    score = getattr(output, 'score', None)
                    if score is not None:
                        # NeMo scores are often log probabilities, not a 0..1
                        # confidence. Keep public callbacks predictable.
                        value = float(score)
                        confidence = value if 0.0 <= value <= 1.0 else float(np.exp(value))
                elif isinstance(output, dict):
                    text = output.get('text', '')
                    if 'logprobs' in output:
                        confidence = self.confidence_calc.calculate(output['logprobs'])
                else:
                    text = str(output)
            
            # Calculate confidence if not available
            if confidence == 0.0 and text:
                confidence = self.confidence_calc.from_text(text)
        
        except Exception:
            pass
        
        return text, confidence
    
    def _process_remaining(self):
        """Process any remaining audio in queue"""
        while not self.processing_queue.empty():
            try:
                item = self.processing_queue.get_nowait()
                if item['type'] == 'final':
                    self._process_final(item['audio'], item['duration'])
            except:
                break
    
    def _save_transcript(self):
        """Save transcript to file"""
        if not self.text_buffer.text:
            return
        
        # Create transcripts directory
        transcripts_dir = Path(self.config.transcripts_dir)
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = transcripts_dir / f"transcript_{timestamp}.txt"
        
        # Save transcript
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.text_buffer.text)
        
        print(f"💾 Transcript saved: {filename}")
    
    def get_current_transcript(self) -> str:
        """Get current full transcript"""
        return self.text_buffer.text
    
    def get_partial_transcript(self) -> str:
        """Get current partial transcript"""
        return self.state.partial_text
    
    def clear_transcript(self):
        """Clear current transcript"""
        self.text_buffer.clear()
        self.state.current_text = ""
        self.state.partial_text = ""
        print("🗑️  Transcript cleared")
    
    def get_statistics(self) -> Dict:
        """Get engine statistics"""
        return {
            **self.stats,
            'audio_stats': self.audio_capture.get_statistics(),
            'current_words': self.state.words_processed
        }
