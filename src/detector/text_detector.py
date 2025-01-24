import cv2
import pytesseract
from src.utils.image_preprocessing import ImagePreprocessor
import numpy as np

class TextDetector:
    @staticmethod
    def extract_number(text: str) -> float:
        # Remove 'BB' or 'bb' from the text
        text = text.upper().replace('BB', '')

        numbers = ''.join(c for c in text if c.isdigit() or c == '.')
        try:
            return float(numbers)
        except ValueError:
            return 0.0

    def detect_text(self, roi: np.ndarray) -> str:

        processed = ImagePreprocessor.preprocess_for_ocr(roi)
        
        custom_config = r'--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.'
        return pytesseract.image_to_string(processed, config=custom_config)


    def detect_value(self, roi: np.ndarray) -> float:
        text = self.detect_text(roi)
        return self.extract_number(text)