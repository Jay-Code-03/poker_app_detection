import cv2
import pytesseract
from src.utils.image_preprocessing import ImagePreprocessor
import numpy as np

class TextDetector:
    @staticmethod
    def extract_number(text: str) -> float:
        # Remove common OCR mistakes and normalize text
        text = text.upper().replace('BB', '').replace('O', '0').replace('I', '1')
        text = ''.join(c for c in text if c.isdigit() or c == '.' or c == ',')
        
        # Handle common decimal formatting
        text = text.replace(',', '.')
        
        # Extract the first valid number
        parts = text.split('.')
        if len(parts) > 2:  # Multiple decimals found
            text = parts[0] + '.' + parts[1]
        
        try:
            return float(text)
        except ValueError:
            return 0.0

    def detect_text(self, roi: np.ndarray) -> str:

        processed = ImagePreprocessor.preprocess_for_ocr(roi)
        
        custom_config = r'--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.'
        return pytesseract.image_to_string(processed, config=custom_config)


    def detect_value(self, roi: np.ndarray) -> float:
        # Try multiple preprocessing approaches
        results = []
        
        # Standard preprocessing
        text1 = self.detect_text(roi, False)
        results.append(self.extract_number(text1))
        
        # Dark background preprocessing
        text2 = self.detect_text(roi, True)
        results.append(self.extract_number(text2))
        
        # Return non-zero result or average if multiple valid results
        valid_results = [r for r in results if r > 0]
        if valid_results:
            return max(valid_results)
        return 0.0