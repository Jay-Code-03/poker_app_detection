import keyboard
from threading import Event

class BotController:
    def __init__(self):
        self.stop_event = Event()
        self._setup_keyboard_hooks()
    
    def _setup_keyboard_hooks(self):
        keyboard.add_hotkey('alt+q', self.stop_bot)
    
    def stop_bot(self):
        print("\nStopping bot...")
        self.stop_event.set()
    
    def should_continue(self):
        return not self.stop_event.is_set()
    
    def cleanup(self):
        keyboard.unhook_all()