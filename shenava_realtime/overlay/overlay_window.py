"""
Overlay window for real-time transcription display
Creates a floating, always-on-top window with transcription text
"""

import tkinter as tk
from tkinter import font as tkfont
import threading
import queue
import time
from typing import Optional, Callable
from pathlib import Path

from ..config import OverlayConfig, OverlayPosition


class OverlayWindow:
    """
    Floating overlay window for transcription display
    
    Features:
    - Always on top
    - Semi-transparent
    - Customizable position and appearance
    - Smooth text updates
    - Auto-hide on silence
    - Click-through mode (optional)
    """
    
    def __init__(self, config: OverlayConfig, ui_queue: Optional[queue.Queue] = None):
        """
        Initialize overlay window
        
        Args:
            config: Overlay configuration
            ui_queue: Optional queue of (method_name, args) tuples drained by
                      the tkinter thread (used for thread-safe cross-thread updates)
        """
        self.config = config
        self.is_visible = False
        self.is_running = False
        self.ui_queue = ui_queue
        
        # Create tkinter window
        self.root = None
        self.text_label = None
        self.confidence_label = None
        
        # Update thread
        self.update_thread: Optional[threading.Thread] = None
        self.update_queue = []
        
        # Auto-hide timer
        self.last_text_time = 0.0
        self.auto_hide_timer = None
        
        # Initialize window
        self._create_window()
    
    def _create_window(self):
        """Create the overlay window"""
        # Create root window in separate thread
        self.root = tk.Tk()
        self.root.title("Shenava ASR Overlay")
        
        # Configure window
        self._configure_window()
        
        # Create widgets
        self._create_widgets()
        
        # Position window
        self._position_window()
        
        # Bind events
        self._bind_events()
        
        # Mark visible: the Tk window is shown on creation, so keep the
        # flag in sync (otherwise update_text/update_partial drop everything)
        self.show()
    
    def _configure_window(self):
        """Configure window properties"""
        # Always on top
        if self.config.always_on_top:
            self.root.attributes('-topmost', True)
        
        # Transparency
        self.root.attributes('-alpha', self.config.opacity)
        
        # Remove window decorations
        if self.config.borderless:
            self.root.overrideredirect(True)
        
        # Set size
        self.root.geometry(f"{self.config.width}x{self.config.height}")
        
        # Set background color
        self.root.configure(bg=self.config.background_color)
        
        # Try to make click-through (Windows only)
        if self.config.click_through and hasattr(self.root, 'attributes'):
            try:
                # This is Windows-specific
                self.root.attributes('-transparentcolor', self.config.background_color)
            except:
                pass
    
    def _create_widgets(self):
        """Create overlay widgets"""
        # Main frame
        self.main_frame = tk.Frame(
            self.root,
            bg=self.config.background_color,
            padx=10,
            pady=5
        )
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Try to load Persian font
        font_family = self.config.font_family
        try:
            # Check if font exists
            available_fonts = tkfont.families()
            if font_family not in available_fonts:
                # Fallback fonts
                for fallback in ['Segoe UI', 'Arial', 'Helvetica']:
                    if fallback in available_fonts:
                        font_family = fallback
                        break
                else:
                    font_family = 'TkDefaultFont'
        except:
            font_family = 'TkDefaultFont'
        
        # Text label
        self.text_label = tk.Label(
            self.main_frame,
            text="",
            font=(font_family, self.config.font_size),
            fg=self.config.text_color,
            bg=self.config.background_color,
            wraplength=self.config.width - 20,
            justify=tk.CENTER,
            anchor=tk.CENTER
        )
        self.text_label.pack(fill=tk.BOTH, expand=True)
        
        # Confidence label (optional)
        if self.config.show_confidence:
            self.confidence_label = tk.Label(
                self.main_frame,
                text="",
                font=(font_family, 10),
                fg='#888888',
                bg=self.config.background_color,
                anchor=tk.S
            )
            self.confidence_label.pack(fill=tk.X)
        
        # Status indicator
        self.status_label = tk.Label(
            self.main_frame,
            text="●",
            font=(font_family, 8),
            fg='#4A9EFF',
            bg=self.config.background_color,
            anchor=tk.E
        )
        self.status_label.place(relx=1.0, rely=0.0, anchor=tk.NE)
    
    def _position_window(self):
        """Position the overlay window"""
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate position based on config
        if self.config.position == OverlayPosition.TOP:
            x = (screen_width - self.config.width) // 2
            y = 50
        
        elif self.config.position == OverlayPosition.BOTTOM:
            x = (screen_width - self.config.width) // 2
            y = screen_height - self.config.height - 100
        
        elif self.config.position == OverlayPosition.TOP_LEFT:
            x = 20
            y = 50
        
        elif self.config.position == OverlayPosition.TOP_RIGHT:
            x = screen_width - self.config.width - 20
            y = 50
        
        elif self.config.position == OverlayPosition.BOTTOM_LEFT:
            x = 20
            y = screen_height - self.config.height - 100
        
        elif self.config.position == OverlayPosition.BOTTOM_RIGHT:
            x = screen_width - self.config.width - 20
            y = screen_height - self.config.height - 100
        
        elif self.config.position == OverlayPosition.CENTER:
            x = (screen_width - self.config.width) // 2
            y = (screen_height - self.config.height) // 2
        
        elif self.config.position == OverlayPosition.CUSTOM:
            x, y = self.config.custom_position
        
        else:
            # Default: bottom center
            x = (screen_width - self.config.width) // 2
            y = screen_height - self.config.height - 100
        
        # Set position
        self.root.geometry(f"{self.config.width}x{self.config.height}+{x}+{y}")
    
    def _bind_events(self):
        """Bind window events"""
        # Make window draggable
        self.root.bind('<Button-1>', self._on_click)
        self.root.bind('<B1-Motion>', self._on_drag)
        
        # Close on Escape
        self.root.bind('<Escape>', lambda e: self.hide())
        
        # Update loop
        self.root.after(50, self._update_loop)
    
    def _on_click(self, event):
        """Handle click event for dragging"""
        self._drag_start_x = event.x
        self._drag_start_y = event.y
    
    def _on_drag(self, event):
        """Handle drag event"""
        x = self.root.winfo_x() + (event.x - self._drag_start_x)
        y = self.root.winfo_y() + (event.y - self._drag_start_y)
        self.root.geometry(f"+{x}+{y}")
    
    def show(self):
        """Show overlay window"""
        if not self.is_visible:
            self.is_visible = True
            self.root.deiconify()
            self.root.lift()
            
            # Restart auto-hide timer
            self._reset_auto_hide()
    
    def hide(self):
        """Hide overlay window"""
        if self.is_visible:
            self.is_visible = False
            self.root.withdraw()
    
    def toggle(self):
        """Toggle overlay visibility"""
        if self.is_visible:
            self.hide()
        else:
            self.show()
    
    def update_text(self, text: str, confidence: float = 0.0):
        """
        Update overlay text
        
        Args:
            text: Transcription text
            confidence: Confidence score
        """
        # Re-show on new speech if auto-hidden
        if not self.is_visible:
            self.show()
        
        # Truncate text if too long
        if len(text) > 200:
            text = "..." + text[-200:]
        
        # Update text
        self.text_label.config(text=text)
        self.last_text_time = time.time()
        
        # Update confidence
        if self.confidence_label and confidence > 0:
            conf_text = f"Confidence: {confidence:.1%}"
            self.confidence_label.config(text=conf_text)
            
            # Color based on confidence
            if confidence > 0.8:
                color = '#4CAF50'  # Green
            elif confidence > 0.5:
                color = '#FFC107'  # Yellow
            else:
                color = '#F44336'  # Red
            
            self.confidence_label.config(fg=color)
        
        # Update status indicator
        if confidence > 0:
            self.status_label.config(fg='#4CAF50')  # Green when active
        else:
            self.status_label.config(fg='#888888')  # Gray when idle
        
        # Reset auto-hide timer
        self._reset_auto_hide()
    
    def update_partial(self, text: str):
        """
        Update with partial transcription
        
        Args:
            text: Partial transcription text
        """
        if not self.config.show_partial:
            return
        
        # Add ellipsis for partial results
        display_text = text + "..."
        self.text_label.config(text=display_text)
        self.last_text_time = time.time()
    
    def set_recording_status(self, is_recording: bool):
        """
        Set recording status indicator
        
        Args:
            is_recording: Whether currently recording
        """
        if is_recording:
            self.status_label.config(fg='#4A9EFF', text='●')
        else:
            self.status_label.config(fg='#888888', text='○')
    
    def _reset_auto_hide(self):
        """Reset auto-hide timer"""
        if self.auto_hide_timer:
            self.root.after_cancel(self.auto_hide_timer)
        
        if self.config.auto_hide_delay > 0:
            self.auto_hide_timer = self.root.after(
                int(self.config.auto_hide_delay * 1000),
                self._auto_hide
            )
    
    def _auto_hide(self):
        """Auto-hide overlay"""
        if self.is_visible:
            # Fade out effect would go here
            self.hide()
    
    def _update_loop(self):
        """Main update loop (runs on the tkinter thread)"""
        # Drain cross-thread update queue
        if self.ui_queue is not None:
            try:
                while True:
                    method_name, args = self.ui_queue.get_nowait()
                    fn = getattr(self, method_name, None)
                    if callable(fn):
                        fn(*args)
            except queue.Empty:
                pass
        
        # Check for auto-hide
        if self.is_visible and self.last_text_time > 0:
            if time.time() - self.last_text_time > self.config.auto_hide_delay:
                self.last_text_time = 0.0  # Only hide once per idle period
                self._auto_hide()
        
        # Schedule next update
        self.root.after(50, self._update_loop)
    
    def run(self):
        """Run the overlay window main loop"""
        self.is_running = True
        self.root.mainloop()
    
    def stop(self):
        """Stop the overlay window"""
        self.is_running = False
        if self.root:
            self.root.quit()
            self.root.destroy()
    
    def set_position(self, x: int, y: int):
        """
        Set custom position
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        self.root.geometry(f"+{x}+{y}")
    
    def set_size(self, width: int, height: int):
        """
        Set window size
        
        Args:
            width: Window width
            height: Window height
        """
        self.root.geometry(f"{width}x{height}")


class OverlayManager:
    """
    Manages overlay window lifecycle and updates
    """
    
    def __init__(self, config: OverlayConfig):
        """
        Initialize overlay manager
        
        Args:
            config: Overlay configuration
        """
        self.config = config
        self.overlay: Optional[OverlayWindow] = None
        self.overlay_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # Cross-thread UI updates are queued and drained on the tkinter thread
        self.ui_queue: queue.Queue = queue.Queue()
        self._last_recording_status: Optional[bool] = None
    
    def start(self):
        """Start overlay in separate thread"""
        if self.is_running:
            return
        
        print("🖥️  Starting overlay window...")
        
        self.is_running = True
        
        # Create and run overlay in separate thread
        self.overlay_thread = threading.Thread(target=self._run_overlay)
        self.overlay_thread.daemon = True
        self.overlay_thread.start()
        
        # Wait for overlay to initialize
        time.sleep(0.5)
        
        print("✅ Overlay window started")
    
    def _run_overlay(self):
        """Run overlay window"""
        try:
            self.overlay = OverlayWindow(self.config, ui_queue=self.ui_queue)
            self.overlay.run()
        except Exception as e:
            print(f"❌ Overlay error: {e}")
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop overlay"""
        if not self.is_running:
            return
        
        print("⏹️  Stopping overlay window...")
        
        self.is_running = False
        
        if self.overlay:
            self.overlay.stop()
        
        if self.overlay_thread:
            self.overlay_thread.join(timeout=2.0)
        
        print("✅ Overlay window stopped")
    
    def update_text(self, text: str, confidence: float = 0.0):
        """
        Update overlay text (thread-safe)
        
        Args:
            text: Transcription text
            confidence: Confidence score
        """
        self.ui_queue.put(('update_text', (text, confidence)))
    
    def update_partial(self, text: str):
        """
        Update partial text (thread-safe)
        
        Args:
            text: Partial transcription
        """
        self.ui_queue.put(('update_partial', (text,)))
    
    def toggle(self):
        """Toggle overlay visibility"""
        self.ui_queue.put(('toggle', ()))
    
    def show(self):
        """Show overlay"""
        self.ui_queue.put(('show', ()))
    
    def hide(self):
        """Hide overlay"""
        self.ui_queue.put(('hide', ()))
    
    def set_recording_status(self, is_recording: bool):
        """
        Set recording status (thread-safe)
        
        Args:
            is_recording: Recording status
        """
        # Skip duplicate updates (main loop pushes this frequently)
        if is_recording == self._last_recording_status:
            return
        self._last_recording_status = is_recording
        self.ui_queue.put(('set_recording_status', (is_recording,)))