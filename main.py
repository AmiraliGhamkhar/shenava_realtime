#!/usr/bin/env python3
"""
Shenava Real-time ASR - Main Application
Real-time Persian speech recognition with overlay and text injection
"""

import sys
import signal
import time
import threading
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Force UTF-8 stdio so Persian transcription text survives Windows console (cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from shenava_realtime.config import (
    AppConfig, ConfigManager, OutputMode
)
from shenava_realtime.realtime_engine import RealtimeASR
from shenava_realtime.overlay.overlay_window import OverlayManager
from shenava_realtime.injector.text_injector import TextInjector
from shenava_realtime.hotkeys.hotkey_manager import HotkeyManager
from shenava_realtime.utils import (
    PerformanceMonitor, 
    WindowManager,
    StateManager
)


class ShenavaApp:
    """
    Main application controller for Shenava Real-time ASR
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize application
        
        Args:
            config: Application configuration
        """
        self.config = config or AppConfig()
        self.is_running = False
        
        # State
        self.state_manager = StateManager()
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize components
        print("🚀 Initializing Shenava Real-time ASR...")
        print("=" * 60)
        
        self._init_components()
        self._setup_callbacks()
        self._setup_hotkeys()
        
        print("=" * 60)
        print("✅ All components initialized")
    
    def _init_components(self):
        """Initialize all components"""
        
        # Initialize ASR engine
        print("\n1️⃣  Initializing ASR Engine...")
        self.asr_engine = RealtimeASR(self.config)
        
        # Initialize overlay
        if self.config.overlay.enabled:
            print("\n2️⃣  Initializing Overlay Window...")
            self.overlay = OverlayManager(self.config.overlay)
        else:
            self.overlay = None
            print("\n2️⃣  Overlay disabled")
        
        # Initialize injector
        if self.config.injector.enabled:
            print("\n3️⃣  Initializing Text Injector...")
            self.injector = TextInjector(self.config.injector)
        else:
            self.injector = None
            print("\n3️⃣  Injector disabled")
        
        # Initialize hotkeys
        print("\n4️⃣  Initializing Hotkeys...")
        self.hotkeys = HotkeyManager(self.config.hotkeys)
    
    def _setup_callbacks(self):
        """Setup component callbacks"""
        
        # ASR callbacks
        self.asr_engine.on_partial_result = self._on_partial_result
        self.asr_engine.on_final_result = self._on_final_result
        self.asr_engine.on_confidence_update = self._on_confidence_update
        
        # Injector callbacks
        if self.injector:
            self.injector.on_injection_complete = self._on_injection_complete
            self.injector.on_injection_failed = self._on_injection_failed
    
    def _setup_hotkeys(self):
        """Setup global hotkeys"""
        self.hotkeys.setup_default_hotkeys(
            toggle_recording=self.toggle_recording,
            toggle_overlay=self.toggle_overlay,
            toggle_injector=self.toggle_injector,
            clear_transcript=self.clear_transcript,
            emergency_stop=self.emergency_stop
        )
        
        print("\n⌨️  Hotkeys:")
        for combo, desc in self.hotkeys.get_bindings().items():
            print(f"   {combo}: {desc}")
    
    def start(self):
        """Start the application"""
        if self.is_running:
            return
        
        print("\n" + "="*60)
        print("🎙️  STARTING SHENAVA REAL-TIME ASR")
        print("="*60)
        
        self.is_running = True
        self.state_manager.set_state("running")
        
        try:
            # Start hotkeys
            self.hotkeys.start()
            
            # Start overlay
            if self.overlay:
                self.overlay.start()
            
            # Start injector
            if self.injector:
                self.injector.start()
            
            # Start ASR engine (this starts audio capture)
            self.asr_engine.start()
            
            # Update overlay status
            if self.overlay:
                self.overlay.set_recording_status(True)
            
            print("\n✅ Application started successfully!")
            print("\n🎤 Listening... Speak now!")
            print("\nPress Ctrl+C to stop\n")
            
            # Main loop
            self._main_loop()
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the application"""
        if not self.is_running:
            return
        
        print("\n" + "="*60)
        print("⏹️  STOPPING SHENAVA REAL-TIME ASR")
        print("="*60)
        
        self.is_running = False
        self.state_manager.set_state("stopping")
        
        # Stop components
        self.asr_engine.stop()
        
        if self.injector:
            self.injector.stop()
        
        if self.overlay:
            self.overlay.stop()
        
        self.hotkeys.stop()
        
        self.state_manager.set_state("stopped")
        
        # Print statistics
        self._print_statistics()
        
        print("\n✅ Application stopped")
    
    def _main_loop(self):
        """Main application loop"""
        while self.is_running:
            try:
                # Update performance metrics
                self._update_performance()
                
                # Update overlay
                if self.overlay:
                    self._update_overlay()
                
                # Sleep to avoid high CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Main loop error: {e}")
                time.sleep(1.0)
    
    def _update_performance(self):
        """Update performance metrics"""
        stats = self.asr_engine.get_statistics()
        
        if stats['total_audio_duration'] > 0:
            rtf = stats['total_processing_time'] / stats['total_audio_duration']
            self.performance_monitor.update_rtf(rtf)
    
    def _update_overlay(self):
        """Update overlay with current state"""
        if not self.overlay:
            return
        
        # Update recording status
        self.overlay.set_recording_status(self.asr_engine.is_running)
    
    # ==================== Callbacks ====================
    
    def _on_partial_result(self, text: str, confidence: float):
        """
        Handle partial transcription result
        
        Args:
            text: Partial transcription
            confidence: Confidence score
        """
        # Update overlay
        if self.overlay and self.config.overlay.show_partial:
            self.overlay.update_partial(text)
    
    def _on_final_result(self, text: str, confidence: float):
        """
        Handle final transcription result
        
        Args:
            text: Final transcription
            confidence: Confidence score
        """
        # Update overlay
        if self.overlay:
            self.overlay.update_text(text, confidence)
        
        # Inject text if enabled
        if self.injector and self.config.output_mode in [
            OutputMode.INJECT_ONLY, 
            OutputMode.BOTH
        ]:
            self.injector.inject_text(text)
        
        # Copy to clipboard if enabled
        if self.config.output_mode == OutputMode.CLIPBOARD:
            self._copy_to_clipboard(text)
    
    def _on_confidence_update(self, confidence: float):
        """
        Handle confidence update
        
        Args:
            confidence: New confidence value
        """
        # Could update UI or trigger alerts
        pass
    
    def _on_injection_complete(self, text: str):
        """
        Handle successful injection
        
        Args:
            text: Injected text
        """
        if self.config.debug:
            print(f"💉 Injected: {text[:50]}...")
    
    def _on_injection_failed(self, text: str):
        """
        Handle failed injection
        
        Args:
            text: Text that failed to inject
        """
        print(f"⚠️  Injection failed: {text[:50]}...")
    
    def _copy_to_clipboard(self, text: str):
        """
        Copy text to clipboard
        
        Args:
            text: Text to copy
        """
        try:
            import pyperclip
            pyperclip.copy(text)
        except:
            pass
    
    # ==================== Hotkey Actions ====================
    
    def toggle_recording(self):
        """Toggle recording on/off"""
        if self.asr_engine.is_running:
            self.asr_engine.stop()
            print("\n⏸️  Recording paused")
            
            if self.overlay:
                self.overlay.set_recording_status(False)
        else:
            self.asr_engine.start()
            print("\n▶️  Recording resumed")
            
            if self.overlay:
                self.overlay.set_recording_status(True)
    
    def toggle_overlay(self):
        """Toggle overlay visibility"""
        if self.overlay:
            self.overlay.toggle()
            visible = getattr(self.overlay, "is_visible", None)
            status = "shown" if visible is not False else "hidden"
            print(f"\n🖥️  Overlay {status}")
    
    def toggle_injector(self):
        """Toggle injector on/off"""
        if self.injector:
            self.injector.toggle()
    
    def clear_transcript(self):
        """Clear current transcript"""
        self.asr_engine.clear_transcript()
        
        if self.overlay:
            self.overlay.update_text("", 0.0)
    
    def emergency_stop(self):
        """Emergency stop"""
        print("\n🚨 EMERGENCY STOP")
        self.stop()
    
    # ==================== Statistics ====================
    
    def _print_statistics(self):
        """Print application statistics"""
        print("\n📊 Application Statistics:")
        print("-" * 40)
        
        # ASR statistics
        asr_stats = self.asr_engine.get_statistics()
        print(f"   Total audio: {asr_stats['total_audio_duration']:.2f}s")
        print(f"   Processing time: {asr_stats['total_processing_time']:.2f}s")
        print(f"   Chunks processed: {asr_stats['num_chunks_processed']}")
        print(f"   Average confidence: {asr_stats['average_confidence']:.3f}")
        
        # Injector statistics
        if self.injector:
            inj_stats = self.injector.get_statistics()
            print(f"\n💉 Injector Statistics:")
            print(f"   Total injections: {inj_stats['total_injections']}")
            print(f"   Successful: {inj_stats['successful']}")
            print(f"   Failed: {inj_stats['failed']}")
        
        # Performance
        perf_stats = self.performance_monitor.get_statistics()
        if 'rtf_values_avg' in perf_stats:
            print(f"\n⚡ Performance:")
            print(f"   Average RTF: {perf_stats['rtf_values_avg']:.3f}")


def signal_handler(signum, frame):
    """Handle system signals"""
    print("\n⚠️  Signal received, stopping...")
    if hasattr(sys, 'app') and sys.app:
        sys.app.stop()
    sys.exit(0)


def main():
    """Main entry point"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run application
    app = ShenavaApp()
    sys.app = app  # Store for signal handler
    
    # Run application
    app.start()


if __name__ == "__main__":
    main()