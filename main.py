import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import time
from src.detector.template_matcher import TemplateMatcher
from src.detector.table_detector import PokerTableDetector
from src.utils.device_connector import DeviceConnector
from src.utils.bot_controller import BotController


class PokerDetectorApp:
    def __init__(self):
        self.device = DeviceConnector.connect_device()
        self.template_matcher = TemplateMatcher('card_templates')
        self.table_detector = PokerTableDetector(self.template_matcher)
        self.bot_controller = BotController()

    def capture_screen(self) -> np.ndarray:
        screenshot_data = self.device.screencap()
        nparr = np.frombuffer(screenshot_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def print_available_actions(self, actions):
        print("\nAvailable Actions:")
        if actions['FOLD']:
            print("- FOLD")
        if actions['CALL']:
            print("- CALL")
        if actions['CHECK']:
            print("- CHECK")
        if actions['R']:
            print(f"- RAISE options: {actions['R']}")
        if actions['B']:
            print(f"- BET options: {actions['B']}")

    def run(self):
        previous_state = None
        
        print("Bot started. Press Ctrl+C to stop.") 
        
        while self.bot_controller.should_continue():
            try:
                screen = self.capture_screen()
                is_hero_turn = self.table_detector.detect_hero_turn(screen)
                
                if is_hero_turn:
                    current_state = self.table_detector.detect_table_state(screen)
                    
                    if self._has_state_changed(previous_state, current_state):
                        print("\n=== Table State ===")
                        print("Hero cards:", [f"{c.rank}{c.suit}" for c in current_state['hero_cards']])
                        print("Community cards:", [f"{c.rank}{c.suit}" for c in current_state['community_cards']])
                        print(f"Hero stack: ${current_state['stacks']['hero']:.2f}")
                        print(f"Villain stack: ${current_state['stacks']['villain']:.2f}")
                        print(f"Hero bet: ${current_state['bets']['hero']:.2f}")
                        print(f"Villain bet: ${current_state['bets']['villain']:.2f}")
                        print(f"Pot size: ${current_state['pot_size']:.2f}")
                        print(f"Button positions: {current_state['button_positions']}")
                        print("================")
                        
                        # Print available actions in a cleaner format
                        self.print_available_actions(current_state['available_actions'])
                        print("================\n")
                        
                        previous_state = current_state
                
                time.sleep(1)
                
            except Exception as e:
                print(f"Error occurred: {e}")
                import traceback
                traceback.print_exc()  # This will print the full error trace
                break
        
        self.cleanup()

    def _has_state_changed(self, previous_state, current_state):
        if previous_state is None:
            return True
            
        # Compare relevant state components
        return (
            previous_state['hero_cards'] != current_state['hero_cards'] or
            previous_state['community_cards'] != current_state['community_cards'] or
            previous_state['bets'] != current_state['bets'] or
            previous_state['pot_size'] != current_state['pot_size'] or
            previous_state['available_actions'] != current_state['available_actions']  # Added this check
        )
    
    def cleanup(self):
        self.bot_controller.cleanup()
        cv2.destroyAllWindows()

def main():
    app = PokerDetectorApp()

    try:
        app.run()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        app.cleanup()

if __name__ == "__main__":
    main()