import pytesseract
from src.utils.image_preprocessing import ImagePreprocessor
import numpy as np

class TextDetector:
    @staticmethod
    def extract_number(text: str) -> float:
        # Remove 'BB' or 'bb' from the text
        text = text.upper().replace('BB', '')

        # Extract numbers and decimal points
        numbers = ''.join(c for c in text if c.isdigit() or c == '.')
        try:
            return float(numbers)
        except ValueError:
            return 0.0

    def detect_text(self, roi: np.ndarray, is_dark_background: bool = True) -> str:
        # Preprocess the image
        processed = ImagePreprocessor.preprocess_for_ocr(roi, is_dark_background)

        # Use Tesseract with custom configuration
        custom_config = r'--psm 7 -c tessedit_char_whitelist=0123456789.'
        text = pytesseract.image_to_string(processed, config=custom_config)

        return text

    def detect_value(self, roi: np.ndarray, is_dark_background: bool = True) -> float:
        text = self.detect_text(roi, is_dark_background)
        return self.extract_number(text)