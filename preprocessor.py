"""
Image Preprocessing: Clean and enhance images for better OCR
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocess images to improve OCR accuracy"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
    def assess_quality(self, image: np.ndarray) -> Dict:
        """Assess image quality metrics"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate metrics
        brightness = np.mean(gray)
        contrast = gray.std()
        
        # Laplacian variance (sharpness)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        return {
            "brightness": float(brightness),
            "contrast": float(contrast),
            "sharpness": float(sharpness),
            "needs_enhancement": brightness < 50 or brightness > 200 or contrast < 30
        }
    
    def deskew(self, image: np.ndarray) -> np.ndarray:
        """Detect and correct skew/rotation"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is not None:
            # Calculate average angle
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta) - 90
                if -45 < angle < 45:
                    angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
                
                # Only rotate if angle is significant
                if abs(median_angle) > 0.5:
                    logger.info(f"Deskewing by {median_angle:.2f} degrees")
                    (h, w) = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    rotated = cv2.warpAffine(
                        image, M, (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE
                    )
                    return rotated
        
        return image
    
    def denoise(self, image: np.ndarray, strength: int = 10) -> np.ndarray:
        """Remove noise from image"""
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(
                image, None, strength, strength, 7, 21
            )
        else:
            denoised = cv2.fastNlMeansDenoising(
                image, None, strength, 7, 21
            )
        return denoised
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def binarize(self, image: np.ndarray) -> np.ndarray:
        """Convert to binary using adaptive thresholding"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        return binary
    
    def preprocess(
        self,
        image_path: Path,
        output_path: Path = None
    ) -> Tuple[np.ndarray, Dict]:
        """Complete preprocessing pipeline"""
        logger.info(f"Preprocessing {image_path.name}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Assess original quality
        original_quality = self.assess_quality(image)
        
        processed = image.copy()
        steps_applied = []
        
        # Apply preprocessing steps
        if self.config.get("deskew", True):
            processed = self.deskew(processed)
            steps_applied.append("deskew")
        
        if self.config.get("denoise", True) and original_quality['sharpness'] < 100:
            processed = self.denoise(processed, strength=10)
            steps_applied.append("denoise")
        
        if self.config.get("enhance_contrast", True) and original_quality['needs_enhancement']:
            processed = self.enhance_contrast(processed)
            steps_applied.append("enhance_contrast")
        
        if self.config.get("binarize", True):
            processed = self.binarize(processed)
            steps_applied.append("binarize")
        
        # Assess final quality
        final_quality = self.assess_quality(processed)
        
        # Save processed image
        if output_path:
            cv2.imwrite(str(output_path), processed)
        
        metrics = {
            "original_quality": original_quality,
            "final_quality": final_quality,
            "steps_applied": steps_applied
        }
        
        return processed, metrics
