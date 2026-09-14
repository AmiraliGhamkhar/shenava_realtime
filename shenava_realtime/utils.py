"""
Utility classes and functions for Shenava Real-time ASR
"""

import time
import threading
from typing import Optional, List, Dict, Any, Callable
from collections import deque
import numpy as np
import torch


class TextBuffer:
    """
    Thread-safe text buffer for accumulating transcriptions
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize text buffer
        
        Args:
            max_size: Maximum buffer size (characters)
        """
        self.max_size = max_size
        self.text = ""
        self.lock = threading.Lock()
        self.sentences = deque(maxlen=100)
    
    def add_text(self, text: str):
        """
        Add text to buffer
        
        Args:
            text: Text to add
        """
        with self.lock:
            self.text += " " + text if self.text else text
            
            # Trim if too long
            if len(self.text) > self.max_size:
                # Keep last portion
                self.text = self.text[-self.max_size:]
            
            # Add to sentences
            if text.strip():
                self.sentences.append(text.strip())
    
    def get_text(self) -> str:
        """
        Get current text
        
        Returns:
            Current text
        """
        with self.lock:
            return self.text
    
    def get_last_n(self, n: int) -> List[str]:
        """
        Get last N sentences
        
        Args:
            n: Number of sentences
            
        Returns:
            List of sentences
        """
        with self.lock:
            return list(self.sentences)[-n:]
    
    def clear(self):
        """Clear buffer"""
        with self.lock:
            self.text = ""
            self.sentences.clear()
    
    def save_to_file(self, filepath: str):
        """
        Save text to file
        
        Args:
            filepath: Output file path
        """
        with self.lock:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.text)


class ConfidenceCalculator:
    """
    Calculates confidence scores for transcriptions
    """
    
    def __init__(self):
        """Initialize confidence calculator"""
        self.history = deque(maxlen=100)
    
    def calculate(self, logprobs: Any) -> float:
        """
        Calculate confidence from log probabilities
        
        Args:
            logprobs: Log probabilities from model
            
        Returns:
            Confidence score (0-1)
        """
        try:
            if isinstance(logprobs, torch.Tensor):
                # Convert to probabilities
                probs = torch.exp(logprobs)
                
                # Get max probability per timestep
                max_probs, _ = torch.max(probs, dim=-1)
                
                # Calculate mean confidence
                confidence = max_probs.mean().item()
                
                self.history.append(confidence)
                return float(confidence)
            
            elif isinstance(logprobs, np.ndarray):
                probs = np.exp(logprobs)
                max_probs = np.max(probs, axis=-1)
                confidence = np.mean(max_probs)
                
                self.history.append(confidence)
                return float(confidence)
        
        except Exception:
            pass
        
        return 0.5
    
    def from_text(self, text: str) -> float:
        """
        Estimate confidence from text quality
        
        Args:
            text: Transcribed text
            
        Returns:
            Estimated confidence
        """
        if not text:
            return 0.0
        
        score = 1.0
        
        # Check for repeated words (lower confidence)
        words = text.split()
        repetitions = sum(1 for i in range(1, len(words)) if words[i] == words[i-1])
        score -= repetitions * 0.1
        
        # Very short text (lower confidence)
        if len(words) < 2:
            score -= 0.2
        
        # Check for special tokens
        if '[UNK]' in text or '...' in text:
            score -= 0.3
        
        # Ensure valid range
        score = max(0.0, min(1.0, score))
        
        self.history.append(score)
        return score
    
    def get_average(self) -> float:
        """
        Get average confidence
        
        Returns:
            Average confidence
        """
        if self.history:
            return float(np.mean(list(self.history)))
        return 0.0


class AudioLevelMeter:
    """
    Real-time audio level meter
    """
    
    def __init__(self, window_size: int = 50):
        """
        Initialize audio level meter
        
        Args:
            window_size: Number of samples to average
        """
        self.window_size = window_size
        self.levels = deque(maxlen=window_size)
        self.peak_level = 0.0
    
    def update(self, audio_data: np.ndarray) -> float:
        """
        Update with audio data
        
        Args:
            audio_data: Audio data
            
        Returns:
            Current level (RMS)
        """
        if len(audio_data) == 0:
            return 0.0
        
        # Calculate RMS
        rms = float(np.sqrt(np.mean(audio_data ** 2)))
        
        # Update history
        self.levels.append(rms)
        
        # Update peak
        self.peak_level = max(self.peak_level, rms)
        
        return rms
    
    def get_level(self) -> float:
        """
        Get current average level
        
        Returns:
            Average level
        """
        if self.levels:
            return float(np.mean(list(self.levels)))
        return 0.0
    
    def get_peak(self) -> float:
        """
        Get peak level
        
        Returns:
            Peak level
        """
        return self.peak_level
    
    def reset_peak(self):
        """Reset peak level"""
        self.peak_level = 0.0


class PerformanceMonitor:
    """
    Monitors performance metrics
    """
    
    def __init__(self):
        """Initialize performance monitor"""
        self.metrics = {
            'processing_times': deque(maxlen=100),
            'rtf_values': deque(maxlen=100),
            'memory_usage': deque(maxlen=100)
        }
    
    def update_processing_time(self, time: float):
        """
        Update processing time
        
        Args:
            time: Processing time in seconds
        """
        self.metrics['processing_times'].append(time)
    
    def update_rtf(self, rtf: float):
        """
        Update real-time factor
        
        Args:
            rtf: Real-time factor
        """
        self.metrics['rtf_values'].append(rtf)
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get performance statistics
        
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        for name, values in self.metrics.items():
            if values:
                vals = list(values)
                stats[f'{name}_avg'] = float(np.mean(vals))
                stats[f'{name}_std'] = float(np.std(vals))
                stats[f'{name}_min'] = float(np.min(vals))
                stats[f'{name}_max'] = float(np.max(vals))
        
        return stats


class WindowManager:
    """
    Manages window focus and detection
    """
    
    @staticmethod
    def get_active_window_title() -> Optional[str]:
        """
        Get active window title
        
        Returns:
            Window title or None
        """
        try:
            import pygetwindow as gw
            window = gw.getActiveWindow()
            return window.title if window else None
        except:
            return None
    
    @staticmethod
    def get_active_window_process() -> Optional[str]:
        """
        Get active window process name
        
        Returns:
            Process name or None
        """
        try:
            import psutil
            import win32process
            import win32gui
            
            hwnd = win32gui.GetForegroundWindow()
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            process = psutil.Process(pid)
            return process.name()
        except:
            return None
    
    @staticmethod
    def focus_window(title: str) -> bool:
        """
        Focus window by title
        
        Args:
            title: Window title
            
        Returns:
            True if successful
        """
        try:
            import pygetwindow as gw
            windows = gw.getWindowsWithTitle(title)
            if windows:
                window = windows[0]
                if window.isMinimized:
                    window.restore()
                window.activate()
                return True
        except:
            pass
        return False
    
    @staticmethod
    def list_windows() -> List[str]:
        """
        List all window titles
        
        Returns:
            List of window titles
        """
        try:
            import pygetwindow as gw
            return [w.title for w in gw.getAllWindows() if w.title]
        except:
            return []


class StateManager:
    """
    Manages application state transitions
    """
    
    def __init__(self):
        """Initialize state manager"""
        self.state = "idle"
        self.listeners = []
    
    def set_state(self, new_state: str):
        """
        Set new state
        
        Args:
            new_state: New state
        """
        old_state = self.state
        self.state = new_state
        
        # Notify listeners
        for listener in self.listeners:
            try:
                listener(old_state, new_state)
            except:
                pass
    
    def get_state(self) -> str:
        """
        Get current state
        
        Returns:
            Current state
        """
        return self.state
    
    def add_listener(self, listener: Callable):
        """
        Add state change listener
        
        Args:
            listener: Callback function
        """
        self.listeners.append(listener)