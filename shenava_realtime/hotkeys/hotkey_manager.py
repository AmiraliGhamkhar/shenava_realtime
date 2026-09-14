"""
Global hotkey manager for Shenava Real-time ASR
"""

import threading
import time
from typing import Callable, Optional, Dict, List
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")

from ..config import HotkeyConfig


@dataclass
class HotkeyBinding:
    """Hotkey binding"""
    key_combo: str
    callback: Callable
    description: str
    is_active: bool = True


class HotkeyManager:
    """
    Manages global hotkeys for the application
    
    Features:
    - Global hotkey registration
    - Thread-safe callbacks
    - Hotkey conflict detection
    - Dynamic binding/unbinding
    """
    
    def __init__(self, config: HotkeyConfig):
        """
        Initialize hotkey manager
        
        Args:
            config: Hotkey configuration
        """
        self.config = config
        self.is_running = False
        self.bindings: Dict[str, HotkeyBinding] = {}
        
        # Track currently pressed keys for modifier detection
        self._pressed_keys = set()
        
        # Initialize keyboard listener
        self._init_listener()
        
        # Statistics
        self.stats = {
            'total_presses': 0,
            'successful_triggers': 0,
            'failed_triggers': 0
        }
    
    def _init_listener(self):
        """Initialize keyboard listener"""
        try:
            from pynput import keyboard
            self.keyboard = keyboard
            self.listener = None
            print("✅ Hotkey manager initialized (pynput)")
        except ImportError:
            print("⚠️  pynput not available. Install: pip install pynput")
            print("   Hotkeys will not work without pynput")
            self.keyboard = None
            self.listener = None
    
    def start(self):
        """Start hotkey listener"""
        if not self.keyboard or self.is_running:
            return
        
        print("⌨️  Starting hotkey listener...")
        
        self.is_running = True
        
        # Create and start listener
        self.listener = self.keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        self.listener.start()
        
        print("✅ Hotkey listener started")
    
    def stop(self):
        """Stop hotkey listener"""
        if not self.is_running:
            return
        
        print("⏹️  Stopping hotkey listener...")
        
        self.is_running = False
        
        if self.listener:
            self.listener.stop()
        
        print("✅ Hotkey listener stopped")
    
    def bind(self, key_combo: str, callback: Callable, description: str = ""):
        """
        Bind a hotkey
        
        Args:
            key_combo: Key combination (e.g., "ctrl+alt+r")
            callback: Function to call
            description: Description of the hotkey
        """
        # Normalize key combo
        key_combo = self._normalize_key_combo(key_combo)
        
        # Check for conflicts
        if key_combo in self.bindings:
            print(f"⚠️  Hotkey already bound: {key_combo}")
            print(f"   Overwriting: {self.bindings[key_combo].description}")
        
        # Create binding
        binding = HotkeyBinding(
            key_combo=key_combo,
            callback=callback,
            description=description
        )
        
        self.bindings[key_combo] = binding
        
        print(f"⌨️  Bound: {key_combo} -> {description or callback.__name__}")
    
    def unbind(self, key_combo: str):
        """
        Unbind a hotkey
        
        Args:
            key_combo: Key combination to unbind
        """
        key_combo = self._normalize_key_combo(key_combo)
        
        if key_combo in self.bindings:
            del self.bindings[key_combo]
            print(f"⌨️  Unbound: {key_combo}")
    
    def _normalize_key_combo(self, key_combo: str) -> str:
        """
        Normalize key combination string
        
        Args:
            key_combo: Key combination string
            
        Returns:
            Normalized key combination
        """
        # Split by '+' and normalize
        parts = key_combo.lower().split('+')
        parts = [p.strip() for p in parts if p.strip()]
        
        # Sort modifiers for consistent ordering
        modifiers = ['ctrl', 'alt', 'shift', 'cmd', 'win']
        mod_parts = [p for p in parts if p in modifiers]
        key_parts = [p for p in parts if p not in modifiers]
        
        # Sort modifiers in standard order
        mod_parts.sort(key=lambda x: modifiers.index(x))
        
        # Combine
        normalized = '+'.join(mod_parts + key_parts)
        
        return normalized
    
    def _on_key_press(self, key):
        """
        Handle key press event
        
        Args:
            key: Pressed key
        """
        try:
            self._pressed_keys.add(key)
            
            # Get current modifiers
            current_mods = self._get_current_modifiers()
            
            # Get key name
            key_name = self._get_key_name(key)
            
            if not key_name:
                return
            
            # Skip if the key itself is a modifier (e.g. pressing ctrl alone)
            if key_name in ('ctrl', 'alt', 'shift', 'cmd', 'win', 'ctrl_l', 'ctrl_r',
                            'alt_l', 'alt_r', 'alt_gr', 'shift_l', 'shift_r', 'cmd_l', 'cmd_r'):
                return
            
            # Build key combo
            key_combo_parts = current_mods + [key_name]
            key_combo = '+'.join(key_combo_parts)
            
            # Check bindings
            if key_combo in self.bindings:
                binding = self.bindings[key_combo]
                if binding.is_active:
                    # Update statistics
                    self.stats['total_presses'] += 1
                    
                    # Execute callback in separate thread
                    threading.Thread(
                        target=self._execute_callback,
                        args=(binding,),
                        daemon=True
                    ).start()
        
        except Exception as e:
            print(f"⚠️  Hotkey error: {e}")
    
    def _on_key_release(self, key):
        """
        Handle key release event
        
        Args:
            key: Released key
        """
        self._pressed_keys.discard(key)
    
    def _get_current_modifiers(self) -> List[str]:
        """
        Get currently pressed modifiers from the tracked pressed-key set
        
        Returns:
            List of modifier names
        """
        modifiers = []
        
        try:
            Key = self.keyboard.Key
            
            ctrl_keys = {Key.ctrl, Key.ctrl_l, Key.ctrl_r}
            alt_keys = {Key.alt, Key.alt_l, Key.alt_r, Key.alt_gr}
            shift_keys = {Key.shift, Key.shift_l, Key.shift_r}
            
            if self._pressed_keys & ctrl_keys:
                modifiers.append('ctrl')
            if self._pressed_keys & alt_keys:
                modifiers.append('alt')
            if self._pressed_keys & shift_keys:
                modifiers.append('shift')
        except Exception:
            pass
        
        return modifiers
    
    def _get_key_name(self, key) -> Optional[str]:
        """
        Get key name from key object
        
        Args:
            key: Key object
            
        Returns:
            Key name or None
        """
        try:
            if hasattr(key, 'char') and key.char is not None:
                char = key.char
                # With ctrl held, pynput reports control characters (\x01-\x1a);
                # map them back to letters (\x12 -> 'r')
                if len(char) == 1 and '\x01' <= char <= '\x1a':
                    char = chr(ord(char) + 0x40)
                return char.lower()
            elif hasattr(key, 'name'):
                return key.name.lower()
        except:
            pass
        
        return None
    
    def _execute_callback(self, binding: HotkeyBinding):
        """
        Execute hotkey callback
        
        Args:
            binding: Hotkey binding
        """
        try:
            binding.callback()
            self.stats['successful_triggers'] += 1
        except Exception as e:
            print(f"❌ Hotkey callback failed: {e}")
            self.stats['failed_triggers'] += 1
    
    def setup_default_hotkeys(
        self,
        toggle_recording: Callable,
        toggle_overlay: Callable,
        toggle_injector: Callable,
        clear_transcript: Callable,
        emergency_stop: Callable
    ):
        """
        Setup default hotkeys from configuration
        
        Args:
            toggle_recording: Toggle recording callback
            toggle_overlay: Toggle overlay callback
            toggle_injector: Toggle injector callback
            clear_transcript: Clear transcript callback
            emergency_stop: Emergency stop callback
        """
        # Bind hotkeys from config
        self.bind(
            self.config.toggle_recording,
            toggle_recording,
            "Toggle recording"
        )
        
        self.bind(
            self.config.toggle_overlay,
            toggle_overlay,
            "Toggle overlay"
        )
        
        self.bind(
            self.config.toggle_injector,
            toggle_injector,
            "Toggle injector"
        )
        
        self.bind(
            self.config.clear_transcript,
            clear_transcript,
            "Clear transcript"
        )
        
        self.bind(
            self.config.emergency_stop,
            emergency_stop,
            "Emergency stop"
        )
    
    def get_bindings(self) -> Dict[str, str]:
        """
        Get all bindings
        
        Returns:
            Dictionary of key_combo -> description
        """
        return {
            combo: binding.description
            for combo, binding in self.bindings.items()
        }
    
    def get_statistics(self) -> Dict:
        """Get hotkey statistics"""
        return self.stats.copy()