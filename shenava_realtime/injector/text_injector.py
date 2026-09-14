"""
Text injector for injecting transcribed text into other applications
Supports multiple injection methods
"""

import time
import threading
import queue
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings("ignore")

from ..config import InjectorConfig, InjectorMode


class InjectionResult(Enum):
    """Injection result status"""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRY = "retry"


@dataclass
class InjectionTask:
    """Text injection task"""
    text: str
    timestamp: float
    priority: int = 0
    retries: int = 0
    max_retries: int = 3


class TextInjector:
    """
    Injects transcribed text into other applications
    
    Methods:
    1. Keyboard simulation - Simulates typing
    2. Clipboard paste - Copies to clipboard and pastes
    3. Direct input - Windows-specific direct input
    
    Features:
    - Target window selection
    - Injection queue
    - Retry mechanism
    - Delay control
    """
    
    def __init__(self, config: InjectorConfig):
        """
        Initialize text injector
        
        Args:
            config: Injector configuration
        """
        self.config = config
        self.is_running = False
        self.is_active = config.enabled
        
        # Injection queue
        self.injection_queue = queue.Queue()
        self.injection_thread: Optional[threading.Thread] = None
        
        # Initialize injectors based on mode
        self._init_injectors()
        
        # Statistics
        self.stats = {
            'total_injections': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'total_characters': 0
        }
        
        # Callbacks
        self.on_injection_complete: Optional[Callable] = None
        self.on_injection_failed: Optional[Callable] = None
    
    def _init_injectors(self):
        """Initialize injection methods"""
        self.keyboard_injector = None
        self.clipboard_injector = None
        
        if self.config.mode == InjectorMode.KEYBOARD_SIMULATION:
            self._init_keyboard_injector()
        elif self.config.mode == InjectorMode.CLIPBOARD_PASTE:
            self._init_clipboard_injector()
        elif self.config.mode == InjectorMode.DIRECT_INPUT:
            self._init_direct_input()
    
    def _init_keyboard_injector(self):
        """Initialize keyboard simulation injector"""
        try:
            import pyautogui
            pyautogui.FAILSAFE = True  # Move mouse to corner to abort
            self.keyboard_injector = pyautogui
            print("✅ Keyboard injector initialized (pyautogui)")
        except ImportError:
            try:
                import pynput
                self.keyboard_injector = self._PynputInjector()
                print("✅ Keyboard injector initialized (pynput)")
            except ImportError:
                print("⚠️  No keyboard injector available")
                print("   Install: pip install pyautogui OR pip install pynput")
    
    def _init_clipboard_injector(self):
        """Initialize clipboard injector"""
        try:
            import pyperclip
            self.clipboard_injector = pyperclip
            print("✅ Clipboard injector initialized")
        except ImportError:
            print("⚠️  Clipboard injector not available")
            print("   Install: pip install pyperclip")
    
    def _init_direct_input(self):
        """Initialize direct input injector (Windows only)"""
        try:
            import win32api
            import win32con
            self.direct_input = (win32api, win32con)
            print("✅ Direct input injector initialized (Windows)")
        except ImportError:
            print("⚠️  Direct input not available (Windows only)")
    
    class _PynputInjector:
        """Wrapper for pynput keyboard control"""
        
        def __init__(self):
            from pynput.keyboard import Controller, Key
            self.keyboard = Controller()
            self.Key = Key
        
        def typewrite(self, text: str, interval: float = 0.01):
            """Type text character by character"""
            for char in text:
                self.keyboard.type(char)
                if interval > 0:
                    time.sleep(interval)
        
        def press(self, key):
            """Press a key"""
            self.keyboard.press(key)
        
        def release(self, key):
            """Release a key"""
            self.keyboard.release(key)
    
    def start(self):
        """Start injector"""
        if self.is_running:
            return
        
        print("💉 Starting text injector...")
        
        self.is_running = True
        
        # Start injection thread
        self.injection_thread = threading.Thread(target=self._injection_loop)
        self.injection_thread.daemon = True
        self.injection_thread.start()
        
        print("✅ Text injector started")
    
    def stop(self):
        """Stop injector"""
        if not self.is_running:
            return
        
        print("⏹️  Stopping text injector...")
        
        self.is_running = False
        
        if self.injection_thread:
            self.injection_thread.join(timeout=2.0)
        
        print("✅ Text injector stopped")
    
    def inject_text(self, text: str):
        """
        Queue text for injection
        
        Args:
            text: Text to inject
        """
        if not self.is_active or not text.strip():
            return
        
        task = InjectionTask(
            text=text,
            timestamp=time.time()
        )
        
        self.injection_queue.put(task)
    
    def inject_immediate(self, text: str):
        """
        Inject text immediately (bypasses queue)
        
        Args:
            text: Text to inject
        """
        self._perform_injection(text)
    
    def _injection_loop(self):
        """Main injection loop"""
        while self.is_running:
            try:
                # Get task from queue
                task = self.injection_queue.get(timeout=0.1)
                
                # Perform injection
                result = self._perform_injection(task.text)
                
                # Handle retries
                if result == InjectionResult.RETRY and task.retries < task.max_retries:
                    task.retries += 1
                    time.sleep(0.5)  # Wait before retry
                    self.injection_queue.put(task)
                elif result == InjectionResult.FAILED:
                    if self.on_injection_failed:
                        self.on_injection_failed(task.text)
                
                # Mark task as done
                self.injection_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Injection error: {e}")
    
    def _perform_injection(self, text: str) -> InjectionResult:
        """
        Perform text injection
        
        Args:
            text: Text to inject
            
        Returns:
            InjectionResult status
        """
        if not self.is_active:
            return InjectionResult.SKIPPED
        
        try:
            # Focus target window if specified
            if self.config.target_window:
                self._focus_window(self.config.target_window)
            
            # Wait for focus
            time.sleep(0.1)
            
            # Perform injection based on mode
            if self.config.mode == InjectorMode.KEYBOARD_SIMULATION:
                return self._inject_keyboard(text)
            elif self.config.mode == InjectorMode.CLIPBOARD_PASTE:
                return self._inject_clipboard(text)
            elif self.config.mode == InjectorMode.DIRECT_INPUT:
                return self._inject_direct(text)
            
            return InjectionResult.FAILED
            
        except Exception as e:
            print(f"❌ Injection failed: {e}")
            return InjectionResult.FAILED
    
    def _inject_keyboard(self, text: str) -> InjectionResult:
        """
        Inject text using keyboard simulation
        
        Args:
            text: Text to inject
            
        Returns:
            InjectionResult status
        """
        if not self.keyboard_injector:
            return InjectionResult.FAILED
        
        try:
            # Handle Persian text
            if self.config.use_unicode:
                # Use Unicode input
                self._inject_unicode_text(text)
            else:
                # Use regular typing
                if hasattr(self.keyboard_injector, 'typewrite'):
                    self.keyboard_injector.typewrite(
                        text,
                        interval=self.config.delay_between_keys
                    )
                else:
                    # pynput
                    self.keyboard_injector.typewrite(
                        text,
                        interval=self.config.delay_between_keys
                    )
            
            # Add space after if configured
            if self.config.send_space_after and not text.endswith(' '):
                if hasattr(self.keyboard_injector, 'press'):
                    self.keyboard_injector.press(' ')
                    self.keyboard_injector.release(' ')
                else:
                    self.keyboard_injector.keyboard.type(' ')
            
            # Add enter after if configured
            if self.config.send_enter_after:
                if hasattr(self.keyboard_injector, 'press'):
                    self.keyboard_injector.press('enter')
                    self.keyboard_injector.release('enter')
                else:
                    self.keyboard_injector.keyboard.press(self.keyboard_injector.Key.enter)
                    self.keyboard_injector.keyboard.release(self.keyboard_injector.Key.enter)
            
            # Update statistics
            self.stats['total_injections'] += 1
            self.stats['successful'] += 1
            self.stats['total_characters'] += len(text)
            
            return InjectionResult.SUCCESS
            
        except Exception as e:
            print(f"❌ Keyboard injection failed: {e}")
            self.stats['failed'] += 1
            return InjectionResult.FAILED
    
    def _inject_unicode_text(self, text: str):
        """
        Inject Unicode text (for Persian characters)
        
        Args:
            text: Text to inject
        """
        # pyautogui supports Unicode directly
        if hasattr(self.keyboard_injector, 'typewrite'):
            self.keyboard_injector.typewrite(
                text,
                interval=self.config.delay_between_keys
            )
        elif hasattr(self.keyboard_injector, 'keyboard'):
            # pynput
            for char in text:
                self.keyboard_injector.keyboard.type(char)
                if self.config.delay_between_keys > 0:
                    time.sleep(self.config.delay_between_keys)
    
    def _inject_clipboard(self, text: str) -> InjectionResult:
        """
        Inject text using clipboard
        
        Args:
            text: Text to inject
            
        Returns:
            InjectionResult status
        """
        if not self.clipboard_injector:
            return InjectionResult.FAILED
        
        try:
            # Copy to clipboard
            self.clipboard_injector.copy(text)
            
            # Small delay
            time.sleep(0.05)
            
            # Simulate Ctrl+V
            if hasattr(self.keyboard_injector, 'hotkey'):
                self.keyboard_injector.hotkey('ctrl', 'v')
            elif hasattr(self.keyboard_injector, 'keyboard'):
                from pynput.keyboard import Key, Controller
                keyboard = Controller()
                keyboard.press(Key.ctrl)
                keyboard.press('v')
                keyboard.release('v')
                keyboard.release(Key.ctrl)
            
            # Update statistics
            self.stats['total_injections'] += 1
            self.stats['successful'] += 1
            self.stats['total_characters'] += len(text)
            
            return InjectionResult.SUCCESS
            
        except Exception as e:
            print(f"❌ Clipboard injection failed: {e}")
            self.stats['failed'] += 1
            return InjectionResult.FAILED
    
    def _inject_direct(self, text: str) -> InjectionResult:
        """
        Inject text using Windows direct input
        
        Args:
            text: Text to inject
            
        Returns:
            InjectionResult status
        """
        # This would use Windows API for direct input
        # Implementation would be Windows-specific
        return InjectionResult.FAILED
    
    def _focus_window(self, window_title: str):
        """
        Focus a specific window
        
        Args:
            window_title: Title of window to focus
        """
        try:
            # Try using pygetwindow (Windows)
            import pygetwindow as gw
            windows = gw.getWindowsWithTitle(window_title)
            if windows:
                window = windows[0]
                if window.isMinimized:
                    window.restore()
                window.activate()
                time.sleep(0.2)
        except ImportError:
            # Try using pyautogui
            try:
                import pyautogui
                # This is limited
                pass
            except:
                pass
        except Exception as e:
            print(f"⚠️  Could not focus window: {e}")
    
    def set_target_window(self, window_title: str):
        """
        Set target window for injection
        
        Args:
            window_title: Window title
        """
        self.config.target_window = window_title
        print(f"🎯 Target window set: {window_title}")
    
    def set_mode(self, mode: InjectorMode):
        """
        Set injection mode
        
        Args:
            mode: Injection mode
        """
        self.config.mode = mode
        self._init_injectors()
        print(f"💉 Injection mode: {mode.value}")
    
    def toggle(self):
        """Toggle injector on/off"""
        self.is_active = not self.is_active
        status = "enabled" if self.is_active else "disabled"
        print(f"💉 Injector {status}")
    
    def get_statistics(self) -> Dict:
        """Get injector statistics"""
        return self.stats.copy()
    
    def list_windows(self) -> List[str]:
        """
        List available windows (Windows only)
        
        Returns:
            List of window titles
        """
        try:
            import pygetwindow as gw
            return [w.title for w in gw.getAllWindows() if w.title]
        except:
            return []


class ClipboardManager:
    """
    Manages clipboard operations
    """
    
    def __init__(self):
        """Initialize clipboard manager"""
        try:
            import pyperclip
            self.clipboard = pyperclip
            self.available = True
        except ImportError:
            self.available = False
            print("⚠️  Clipboard not available. Install: pip install pyperclip")
    
    def copy(self, text: str) -> bool:
        """
        Copy text to clipboard
        
        Args:
            text: Text to copy
            
        Returns:
            True if successful
        """
        if not self.available:
            return False
        
        try:
            self.clipboard.copy(text)
            return True
        except:
            return False
    
    def paste(self) -> Optional[str]:
        """
        Get text from clipboard
        
        Returns:
            Clipboard text or None
        """
        if not self.available:
            return None
        
        try:
            return self.clipboard.paste()
        except:
            return None
    
    def clear(self):
        """Clear clipboard"""
        if self.available:
            try:
                self.clipboard.copy("")
            except:
                pass