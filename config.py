"""
Configuration for OCR Document Intelligence System
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path("/home/caio/Music/Unified_multimodal/data").parent
INPUT_DIR = BASE_DIR / "input_pdfs"
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"
LOG_DIR = BASE_DIR / "logs"
DB_DIR = BASE_DIR / "database"

# Create directories if they don't exist
for directory in [INPUT_DIR, OUTPUT_DIR, TEMP_DIR, LOG_DIR, DB_DIR]:
    directory.mkdir(exist_ok=True)
    
for subdir in ["text", "metadata", "quality_reports"]:
    (OUTPUT_DIR / subdir).mkdir(exist_ok=True)
    
for subdir in ["images", "preprocessed"]:
    (TEMP_DIR / subdir).mkdir(exist_ok=True)

# OCR Configuration
OCR_CONFIG = {
    "default_engine": "tesseract",  # or "doctr"
    "tesseract": {
        "lang": "eng",
        "config": "--psm 3 --oem 1"  # PSM 3 = auto, OEM 1 = LSTM
    },
    "doctr": {
        "det_arch": "db_resnet50",
        "reco_arch": "crnn_vgg16_bn",
        "pretrained": True,
        "batch_size": 4
    },
    "dpi": 300,  # Image resolution for PDF rendering
    "min_confidence": 60  # Minimum OCR confidence (0-100)
}

# Preprocessing Configuration
PREPROCESS_CONFIG = {
    "enable": True,
    "deskew": True,
    "denoise": True,
    "enhance_contrast": True,
    "binarize": True
}

# Quality thresholds
QUALITY_THRESHOLDS = {
    "excellent": 90,
    "good": 75,
    "acceptable": 60,
    "poor": 40,
    "unreadable": 0
}

# Database (we'll use SQLite for simplicity)
DATABASE_URL = f"sqlite:///{DB_DIR}/documents.db"

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
