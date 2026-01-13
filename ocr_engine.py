"""
OCR Engine: Unified interface for Tesseract and docTR
"""
import pytesseract
from pathlib import Path
from typing import Dict
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Try to import docTR
try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    DOCTR_AVAILABLE = True
except ImportError:
    DOCTR_AVAILABLE = False
    logger.warning("docTR not available")


class OCRResult:
    """Standardized OCR result format"""
    
    def __init__(self):
        self.text = ""
        self.words = []
        self.lines = []
        self.confidence = 0.0
        self.engine = ""
        
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "words": self.words,
            "lines": self.lines,
            "confidence": self.confidence,
            "engine": self.engine
        }


class TesseractOCR:
    """Tesseract OCR engine wrapper"""
    
    def __init__(self, lang: str = "eng", config: str = "--psm 3 --oem 1"):
        self.lang = lang
        self.config = config
        
    def process(self, image_path: Path) -> OCRResult:
        """Run Tesseract OCR on image"""
        logger.info(f"Running Tesseract on {image_path.name}")
        
        result = OCRResult()
        result.engine = "tesseract"
        
        try:
            image = Image.open(image_path)
            
            # Get detailed data
            data = pytesseract.image_to_data(
                image,
                lang=self.lang,
                config=self.config,
                output_type=pytesseract.Output.DICT
            )
            
            # Get plain text
            result.text = pytesseract.image_to_string(
                image,
                lang=self.lang,
                config=self.config
            )
            
            # Process word-level data
            confidences = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:
                    word_info = {
                        'text': data['text'][i],
                        'bbox': {
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i]
                        },
                        'confidence': float(data['conf'][i]) / 100.0
                    }
                    result.words.append(word_info)
                    confidences.append(float(data['conf'][i]))
            
            # Calculate average confidence
            if confidences:
                result.confidence = sum(confidences) / len(confidences) / 100.0
            
            logger.info(f"Tesseract: {len(result.words)} words, confidence: {result.confidence:.2%}")
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            result.confidence = 0.0
        
        return result


class DocTROCR:
    """docTR OCR engine wrapper"""
    
    def __init__(self, det_arch: str = "db_resnet50", reco_arch: str = "crnn_vgg16_bn"):
        if not DOCTR_AVAILABLE:
            raise ImportError("docTR is not installed")
        
        logger.info(f"Loading docTR model: det={det_arch}, reco={reco_arch}")
        self.model = ocr_predictor(
            det_arch=det_arch,
            reco_arch=reco_arch,
            pretrained=True
        )
        
    def process(self, image_path: Path) -> OCRResult:
        """Run docTR OCR on image"""
        logger.info(f"Running docTR on {image_path.name}")
        
        result = OCRResult()
        result.engine = "doctr"
        
        try:
            doc = DocumentFile.from_images(str(image_path))
            ocr_result = self.model(doc)
            
            full_text = []
            confidences = []
            
            for page in ocr_result.pages:
                page_height, page_width = page.dimensions
                
                for block in page.blocks:
                    for line in block.lines:
                        line_text = []
                        
                        for word in line.words:
                            bbox = {
                                'x': int(word.geometry[0][0] * page_width),
                                'y': int(word.geometry[0][1] * page_height),
                                'width': int((word.geometry[1][0] - word.geometry[0][0]) * page_width),
                                'height': int((word.geometry[1][1] - word.geometry[0][1]) * page_height)
                            }
                            
                            word_info = {
                                'text': word.value,
                                'bbox': bbox,
                                'confidence': float(word.confidence)
                            }
                            
                            result.words.append(word_info)
                            line_text.append(word.value)
                            confidences.append(word.confidence)
                        
                        full_text.append(' '.join(line_text))
            
            result.text = '\n'.join(full_text)
            
            if confidences:
                result.confidence = sum(confidences) / len(confidences)
            
            logger.info(f"docTR: {len(result.words)} words, confidence: {result.confidence:.2%}")
            
        except Exception as e:
            logger.error(f"docTR OCR failed: {e}")
            result.confidence = 0.0
        
        return result


class OCREngine:
    """Unified OCR interface with fallback support"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.default_engine = config.get("default_engine", "tesseract")
        
        # Initialize Tesseract
        self.tesseract = TesseractOCR(
            lang=config["tesseract"]["lang"],
            config=config["tesseract"]["config"]
        )
        
        # Initialize docTR if available
        self.doctr = None
        if DOCTR_AVAILABLE and self.default_engine == "doctr":
            try:
                self.doctr = DocTROCR(
                    det_arch=config["doctr"]["det_arch"],
                    reco_arch=config["doctr"]["reco_arch"]
                )
            except Exception as e:
                logger.warning(f"Could not load docTR: {e}")
                self.default_engine = "tesseract"
    
    def process_with_fallback(self, image_path: Path, engine: str = None) -> OCRResult:
        """Process image with fallback"""
        engine = engine or self.default_engine
        min_confidence = self.config.get("min_confidence", 60) / 100.0
        
        # Try primary engine
        if engine == "doctr" and self.doctr:
            result = self.doctr.process(image_path)
            
            if result.confidence < min_confidence:
                logger.warning(f"docTR confidence too low, trying Tesseract...")
                result_tess = self.tesseract.process(image_path)
                
                if result_tess.confidence > result.confidence:
                    return result_tess
            
            return result
        
        else:  # Tesseract
            result = self.tesseract.process(image_path)
            
            if result.confidence < min_confidence and self.doctr:
                logger.warning(f"Tesseract confidence too low, trying docTR...")
                result_doctr = self.doctr.process(image_path)
                
                if result_doctr.confidence > result.confidence:
                    return result_doctr
            
            return result
