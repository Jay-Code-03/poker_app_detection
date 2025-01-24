import cv2
import numpy as np

class ImagePreprocessor:
    @staticmethod
    def preprocess_for_template(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary

    @staticmethod
    def preprocess_for_ocr(roi: np.ndarray) -> np.ndarray:
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Add white padding around the ROI to improve detection
        padded = cv2.copyMakeBorder(gray, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        
        # Apply threshold to handle both light and dark backgrounds
        _, binary1 = cv2.threshold(padded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, binary2 = cv2.threshold(padded, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Combine both thresholds
        combined = cv2.bitwise_or(binary1, binary2)
        
        # Scale up image with better interpolation
        scaled = cv2.resize(combined, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)
        
        # Remove noise while preserving edges
        denoised = cv2.fastNlMeansDenoising(scaled, None, 10, 7, 21)
        
        return denoised