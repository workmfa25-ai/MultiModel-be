"""
PDF Processing: Extract pages and render as images
"""
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict
import hashlib
import logging

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handle PDF file processing"""
    
    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        
    def calculate_hash(self, filepath: Path) -> str:
        """Calculate SHA256 hash of file for deduplication"""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def validate_pdf(self, filepath: Path) -> Dict:
        """Validate PDF file and extract basic info"""
        try:
            doc = fitz.open(filepath)
            
            info = {
                "valid": True,
                "page_count": len(doc),
                "file_size": filepath.stat().st_size,
                "file_hash": self.calculate_hash(filepath),
                "metadata": doc.metadata,
                "is_encrypted": doc.is_encrypted,
                "has_text": False
            }
            
            # Check if PDF has embedded text (native vs scanned)
            text_chars = 0
            for page in doc:
                text_chars += len(page.get_text())
            
            # If average >100 chars per page, likely has embedded text
            info["has_text"] = (text_chars / len(doc)) > 100
            info["avg_text_per_page"] = text_chars / len(doc)
            
            doc.close()
            return info
            
        except Exception as e:
            logger.error(f"PDF validation failed for {filepath}: {e}")
            return {
                "valid": False,
                "error": str(e)
            }
    
    def extract_pages_as_images(
        self, 
        pdf_path: Path, 
        output_dir: Path,
        pages: List[int] = None
    ) -> List[Path]:
        """
        Convert PDF pages to images
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Where to save images
            pages: List of page numbers to extract (None = all pages)
            
        Returns:
            List of image file paths
        """
        doc = fitz.open(pdf_path)
        image_paths = []
        
        # Determine which pages to process
        if pages is None:
            pages = range(len(doc))
        
        logger.info(f"Extracting {len(pages)} pages from {pdf_path.name}")
        
        for page_num in pages:
            try:
                page = doc[page_num]
                
                # Render page to image at specified DPI
                mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # Save as PNG
                image_filename = f"{pdf_path.stem}_page_{page_num+1:04d}.png"
                image_path = output_dir / image_filename
                
                pix.save(str(image_path))
                image_paths.append(image_path)
                
            except Exception as e:
                logger.error(f"Failed to extract page {page_num+1}: {e}")
                continue
        
        doc.close()
        logger.info(f"Extracted {len(image_paths)} pages successfully")
        return image_paths
    
    def extract_native_text(self, pdf_path: Path) -> Dict[int, str]:
        """
        Extract embedded text from PDF (if available)
        Useful for hybrid PDFs with some native text
        """
        doc = fitz.open(pdf_path)
        text_by_page = {}
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                text_by_page[page_num] = text
        
        doc.close()
        return text_by_page
