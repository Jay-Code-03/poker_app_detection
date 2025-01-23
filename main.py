import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import time
from src.detector.template_matcher import TemplateMatcher
from src.detector.table_detector import PokerTableDetector
from src.utils.device_connector import DeviceConnector


class PokerDetectorApp:
    def __init__(self):
        self.device = DeviceConnector.connect_device()
        self.template_matcher = TemplateMatcher('card_templates')
        self.table_detector = PokerTableDetector(self.template_matcher)

    def capture_screen(self) -> np.ndarray:
        screenshot_data = self.device.screencap()
        nparr = np.frombuffer(screenshot_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def find_coordinates(self):
        screen = self.capture_screen()
        cv2.imwrite("poker_screenshot.png", screen)
        
        window_name = 'Card Coordinate Finder'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                window_width = cv2.getWindowImageRect(window_name)[2]
                window_height = cv2.getWindowImageRect(window_name)[3]
                
                scale_x = screen.shape[1] / window_width
                scale_y = screen.shape[0] / window_height
                
                original_x = int(x * scale_x)
                original_y = int(y * scale_y)
                
                print(f"Coordinates: x={original_x}, y={original_y}")
        
        cv2.setMouseCallback(window_name, mouse_callback)
        
        while True:
            cv2.imshow(window_name, screen)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()

    def run(self):
        while True:
            screen = self.capture_screen()
            state = self.table_detector.detect_table_state(screen)
            
            print("\n=== Table State ===")
            print("Hero cards:", [f"{c.rank}{c.suit}" for c in state['hero_cards']])
            print("Community cards:", [f"{c.rank}{c.suit}" for c in state['community_cards']])
            print(f"Hero stack: ${state['stacks']['hero']:.2f}")
            print(f"Villain stack: ${state['stacks']['villain']:.2f}")
            print(f"Hero bet: ${state['bets']['hero']:.2f}")
            print(f"Villain bet: ${state['bets']['villain']:.2f}")
            print(f"Pot size: ${state['pot_size']:.2f}")
            print(f"Button positions: {state['button_positions']}")
            print(f"Is hero's turn: {state['is_hero_turn']}")
            print("================\n")
            
            time.sleep(3)

def main():
    app = PokerDetectorApp()
    #app.find_coordinates()
    app.run()

if __name__ == "__main__":
    main()