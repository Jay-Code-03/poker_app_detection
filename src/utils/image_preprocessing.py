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
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(gray)
        
        # Remove noise
        denoised = cv2.fastNlMeansDenoising(contrast)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            denoised, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 
            11, 
            2
        )
        
        # Scale up image
        scaled = cv2.resize(binary, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        # Remove small noise
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(scaled, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    @staticmethod
    def preprocess_for_ocr_dark_background(roi: np.ndarray) -> np.ndarray:
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Increase contrast for white text
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        contrast = clahe.apply(gray)
        
        # Invert the image (make text dark and background light)
        inverted = cv2.bitwise_not(contrast)
        
        # Apply mild blur to reduce noise
        blurred = cv2.GaussianBlur(inverted, (3,3), 0)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Scale up image
        scaled = cv2.resize(binary, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        # Clean up noise
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(scaled, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
