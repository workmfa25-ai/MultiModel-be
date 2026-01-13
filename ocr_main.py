"""
Main Pipeline: Process PDFs end-to-end
Complete OCR Document Intelligence System
With Layout Analysis & Named Entity Recognition
CROSS-PLATFORM COMPATIBLE (Windows, Linux, macOS)
"""
import logging
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime
from tqdm import tqdm
import sys
import numpy as np

from config import *
from pdf_processor import PDFProcessor
from preprocessor import ImagePreprocessor
from ocr_engine import OCREngine
from quality_checker import QualityChecker
from layout_analyzer import LayoutAnalyzer, save_structure_visualization
from ner_extractor import NERExtractor, save_entities_report

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_DIR / f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Main document processing pipeline"""
    
    def __init__(self):
        """Initialize all processing components"""
        logger.info("Initializing Document Processor...")
        
        self.pdf_processor = PDFProcessor(dpi=OCR_CONFIG["dpi"])
        self.preprocessor = ImagePreprocessor(PREPROCESS_CONFIG)
        self.ocr_engine = OCREngine(OCR_CONFIG)
        self.quality_checker = QualityChecker(QUALITY_THRESHOLDS)
        
        # CROSS-PLATFORM FIX: Force rule-based layout analysis (no ML models)
        self.layout_analyzer = LayoutAnalyzer(use_ml_model=False)
        
        # Initialize NER extractor
        try:
            # CROSS-PLATFORM FIX: Disable geo-tagging by default (requires internet)
            # Set enable_geocoding=True if you want geo-tagging and have internet
            self.ner_extractor = NERExtractor(
                model_name="en_core_web_sm",
                enable_geocoding=False  # Changed from default to False for offline mode
            )
            logger.info("✓ NER extractor initialized (offline mode)")
        except Exception as e:
            logger.warning(f"NER extractor initialization failed: {e}")
            self.ner_extractor = None
        
        logger.info("✓ All components initialized successfully")
    
    def _sanitize_for_json(self, obj: Any) -> Any:
        """
        Recursively convert non-JSON-serializable objects to JSON-safe formats
        Handles: numpy types, booleans, nested dicts/lists, Path objects, sets
        """
        if isinstance(obj, dict):
            return {key: self._sanitize_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, (datetime,)):
            return obj.isoformat()
        elif obj is None:
            return None
        elif isinstance(obj, str):
            return obj
        else:
            # Try to convert unknown objects to string
            try:
                return str(obj)
            except:
                return None
    
    def process_document(self, pdf_path: Path, max_pages: int = None) -> Dict:
        """
        Process a single PDF document through the entire pipeline
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to process (None = all pages)
            
        Returns:
            Dictionary with processing results and output file paths
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing document: {pdf_path.name}")
        logger.info(f"{'='*80}")
        
        start_time = datetime.now()
        
        try:
            # ============================================================
            # STEP 1: VALIDATE PDF
            # ============================================================
            logger.info("\n[1/7] Validating PDF...")
            pdf_info = self.pdf_processor.validate_pdf(pdf_path)
            
            if not pdf_info["valid"]:
                logger.error(f"PDF validation failed: {pdf_info.get('error')}")
                return {
                    "success": False, 
                    "error": pdf_info.get('error'),
                    "document_name": pdf_path.stem
                }
            
            logger.info(f"✓ Valid PDF")
            logger.info(f"  Pages: {pdf_info['page_count']}")
            logger.info(f"  Size: {pdf_info['file_size'] / 1024 / 1024:.2f} MB")
            logger.info(f"  Has embedded text: {pdf_info['has_text']}")
            logger.info(f"  File hash: {pdf_info['file_hash'][:16]}...")
            
            # Determine pages to process
            total_pages = pdf_info['page_count']
            if max_pages and max_pages < total_pages:
                pages_to_process = list(range(max_pages))
                logger.info(f"  Processing first {max_pages} of {total_pages} pages")
            else:
                pages_to_process = list(range(total_pages))
                logger.info(f"  Processing all {total_pages} pages")
            
            # ============================================================
            # STEP 2: EXTRACT PAGES AS IMAGES
            # ============================================================
            logger.info("\n[2/7] Extracting pages as images...")
            image_paths = self.pdf_processor.extract_pages_as_images(
                pdf_path,
                TEMP_DIR / "images",
                pages=pages_to_process
            )
            
            if not image_paths:
                logger.error("No images extracted from PDF")
                return {
                    "success": False,
                    "error": "Failed to extract pages",
                    "document_name": pdf_path.stem
                }
            
            logger.info(f"✓ Extracted {len(image_paths)} pages to images")
            
            # ============================================================
            # STEP 3: PREPROCESS IMAGES
            # ============================================================
            logger.info("\n[3/7] Preprocessing images...")
            preprocessed_paths = []
            preprocess_metrics = []
            
            for img_path in tqdm(image_paths, desc="Preprocessing", unit="page"):
                output_path = TEMP_DIR / "preprocessed" / img_path.name
                try:
                    _, metrics = self.preprocessor.preprocess(img_path, output_path)
                    preprocessed_paths.append(output_path)
                    preprocess_metrics.append(metrics)
                except Exception as e:
                    logger.warning(f"Preprocessing failed for {img_path.name}: {e}")
                    # Use original image if preprocessing fails
                    preprocessed_paths.append(img_path)
                    preprocess_metrics.append({
                        "error": str(e),
                        "steps_applied": []
                    })
            
            logger.info(f"✓ Preprocessed {len(preprocessed_paths)} images")
            
            # ============================================================
            # STEP 4: RUN OCR
            # ============================================================
            logger.info("\n[4/7] Running OCR...")
            ocr_results = []
            page_reports = []
            
            for page_num, img_path in enumerate(tqdm(preprocessed_paths, desc="OCR Processing", unit="page")):
                try:
                    # Run OCR with fallback
                    result = self.ocr_engine.process_with_fallback(img_path)
                    ocr_results.append(result)
                    
                    # Assess quality
                    quality_report = self.quality_checker.assess_ocr_result(result, page_num + 1)
                    page_reports.append(quality_report)
                    
                    # Log quality for this page
                    status_icon = "✓" if quality_report['quality_level'] in ['excellent', 'good'] else "⚠"
                    logger.debug(
                        f"{status_icon} Page {page_num+1}: {result.confidence:.2%} confidence "
                        f"({quality_report['quality_level']}) - {len(result.words)} words"
                    )
                    
                except Exception as e:
                    logger.error(f"OCR failed for page {page_num+1}: {e}")
                    ocr_results.append(None)
                    page_reports.append({
                        "page": page_num + 1,
                        "error": str(e),
                        "confidence": 0.0,
                        "quality_level": "failed",
                        "word_count": 0,
                        "char_count": 0,
                        "issues": ["ocr_failed"],
                        "requires_review": True,
                        "engine_used": "none"
                    })
            
            successful_pages = len([r for r in ocr_results if r])
            logger.info(f"✓ OCR completed for {successful_pages}/{len(pages_to_process)} pages")
            
            # Get document name early for later use
            doc_name = pdf_path.stem
            
            # Prepare full text for NER
            full_text_parts = []
            for i, result in enumerate(ocr_results):
                if result and result.text:
                    full_text_parts.append(result.text)
            full_text = "\n\n".join(full_text_parts)
            
            # ============================================================
            # STEP 5: ANALYZE DOCUMENT STRUCTURE
            # ============================================================
            logger.info("\n[5/7] Analyzing document structure...")
            
            structure_analysis = None
            structure_json_path = None
            structure_txt_path = None
            toc_path = None
            
            try:
                structure_analysis = self.layout_analyzer.analyze_document(
                    ocr_results,
                    doc_name,
                    image_paths=preprocessed_paths
                )
                
                # Create structure output directory
                structure_dir = OUTPUT_DIR / "structure"
                structure_dir.mkdir(exist_ok=True)
                
                # Save structure JSON
                structure_json_path = structure_dir / f"{doc_name}_structure.json"
                with open(structure_json_path, 'w', encoding='utf-8') as f:
                    json.dump(
                        self._sanitize_for_json(structure_analysis),
                        f,
                        indent=2,
                        ensure_ascii=False
                    )
                
                # Save human-readable visualization
                structure_txt_path = structure_dir / f"{doc_name}_structure.txt"
                save_structure_visualization(structure_analysis, structure_txt_path)
                
                # Extract and save table of contents
                toc = self.layout_analyzer.extract_table_of_contents(structure_analysis['hierarchy'])
                toc_path = structure_dir / f"{doc_name}_toc.json"
                with open(toc_path, 'w', encoding='utf-8') as f:
                    json.dump(self._sanitize_for_json(toc), f, indent=2, ensure_ascii=False)
                
                logger.info(f"  ✓ Document structure analyzed")
                
                # Log statistics
                stats = structure_analysis.get('statistics', {})
                type_dist = stats.get('type_distribution', {})
                logger.info(f"    Sections: {type_dist.get('section', 0)}")
                logger.info(f"    Subsections: {type_dist.get('subsection', 0)}")
                logger.info(f"    Paragraphs: {type_dist.get('paragraph', 0)}")
                logger.info(f"    Tables: {type_dist.get('table', 0)}")
                logger.info(f"    Figures: {type_dist.get('figure', 0)}")
                logger.info(f"    Max depth: {stats.get('max_depth', 0)}")
                
            except Exception as e:
                logger.error(f"  ✗ Structure analysis failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            # ============================================================
            # STEP 6: EXTRACT NAMED ENTITIES
            # ============================================================
            logger.info("\n[6/7] Extracting named entities...")
            
            entities_analysis = None
            entities_json_path = None
            entities_txt_path = None
            
            if self.ner_extractor and full_text:
                try:
                    entities_analysis = self.ner_extractor.extract_document_entities(
                        structure_analysis,
                        full_text
                    )
                    
                    # Create entities output directory
                    entities_dir = OUTPUT_DIR / "entities"
                    entities_dir.mkdir(exist_ok=True)
                    
                    # Save entities JSON
                    entities_json_path = entities_dir / f"{doc_name}_entities.json"
                    with open(entities_json_path, 'w', encoding='utf-8') as f:
                        json.dump(
                            self._sanitize_for_json(entities_analysis),
                            f,
                            indent=2,
                            ensure_ascii=False
                        )
                    
                    # Save human-readable report
                    entities_txt_path = entities_dir / f"{doc_name}_entities.txt"
                    save_entities_report(entities_analysis, entities_txt_path)
                    
                    logger.info(f"  ✓ Entities extracted")
                    
                    # Log statistics
                    stats = entities_analysis.get('statistics', {})
                    logger.info(f"    Total entities: {stats.get('total_entities', 0)}")
                    logger.info(f"    Unique entities: {stats.get('unique_entities', 0)}")
                    logger.info(f"    Entity types: {len(stats.get('entity_types', []))}")
                    
                    # Log language
                    if entities_analysis.get('document_language'):
                        lang = entities_analysis['document_language']
                        logger.info(f"    Language: {lang.get('full_name', 'Unknown')}")
                    
                    # Log geo-tags (may be empty if disabled)
                    geo_tags = entities_analysis.get('geo_tags', {})
                    if geo_tags.get('countries'):
                        logger.info(f"    Countries: {', '.join(geo_tags['countries'][:5])}")
                    
                except Exception as e:
                    logger.error(f"  ✗ Entity extraction failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            else:
                logger.warning(f"  ⚠ Entity extraction skipped (NER not available or no text)")
            
            # ============================================================
            # STEP 7: GENERATE REPORTS AND SAVE RESULTS
            # ============================================================
            logger.info("\n[7/7] Generating reports and saving results...")
            
            # Generate quality report
            quality_report = self.quality_checker.generate_document_report(
                page_reports,
                doc_name
            )
            
            # Save quality report
            quality_report_path = OUTPUT_DIR / "quality_reports" / f"{doc_name}_quality.json"
            self.quality_checker.save_report(quality_report, quality_report_path)
            logger.info(f"  ✓ Quality report saved")
            
            # Save extracted text
            text_output_path = OUTPUT_DIR / "text" / f"{doc_name}.txt"
            full_text_with_headers = []
            
            for i, result in enumerate(ocr_results):
                if result:
                    page_header = f"{'='*80}\nPage {i+1}\n{'='*80}\n"
                    full_text_with_headers.append(page_header + result.text)
                else:
                    full_text_with_headers.append(f"{'='*80}\nPage {i+1}\n{'='*80}\n[OCR FAILED]")
            
            text_output_path.write_text("\n\n".join(full_text_with_headers), encoding='utf-8')
            logger.info(f"  ✓ Text extracted and saved")
            
            # Save detailed word-level data (useful for advanced processing)
            words_data_path = OUTPUT_DIR / "metadata" / f"{doc_name}_words.json"
            words_data = {
                "document_name": doc_name,
                "pages": [
                    {
                        "page_num": i + 1,
                        "words": result.words if result else [],
                        "confidence": result.confidence if result else 0.0,
                        "engine": result.engine if result else "none"
                    }
                    for i, result in enumerate(ocr_results)
                ]
            }
            
            # Sanitize word data before saving
            words_data_sanitized = self._sanitize_for_json(words_data)
            
            with open(words_data_path, 'w', encoding='utf-8') as f:
                json.dump(words_data_sanitized, f, indent=2, ensure_ascii=False)
            logger.info(f"  ✓ Word-level data saved")
            
            # Save comprehensive metadata
            metadata = {
                "document_name": doc_name,
                "original_file": str(pdf_path.absolute()),
                "processed_at": datetime.now().isoformat(),
                "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
                
                "pdf_info": {
                    "page_count": pdf_info['page_count'],
                    "file_size_mb": round(pdf_info['file_size'] / 1024 / 1024, 2),
                    "file_hash": pdf_info['file_hash'],
                    "has_embedded_text": bool(pdf_info['has_text']),
                    "avg_text_per_page": float(pdf_info.get('avg_text_per_page', 0))
                },
                
                "processing_summary": {
                    "pages_processed": len(pages_to_process),
                    "pages_successful": successful_pages,
                    "pages_failed": len(pages_to_process) - successful_pages
                },
                
                "quality_summary": {
                    "avg_confidence": float(quality_report["avg_confidence"]),
                    "overall_quality": quality_report["overall_quality"],
                    "quality_distribution": quality_report["quality_distribution"],
                    "problem_pages": quality_report["problem_pages"],
                    "recommendation": quality_report["recommendation"]
                },
                
                "structure_summary": {
                    "analyzed": structure_analysis is not None,
                    "statistics": structure_analysis.get('statistics', {}) if structure_analysis else {}
                } if structure_analysis else {"analyzed": False},
                
                "entities_summary": {
                    "extracted": entities_analysis is not None,
                    "statistics": entities_analysis.get('statistics', {}) if entities_analysis else {},
                    "language": entities_analysis.get('document_language', {}) if entities_analysis else {},
                    "countries": entities_analysis.get('geo_tags', {}).get('countries', []) if entities_analysis else []
                } if entities_analysis else {"extracted": False},
                
                "configuration": {
                    "ocr_engine": OCR_CONFIG["default_engine"],
                    "dpi": OCR_CONFIG["dpi"],
                    "preprocessing_enabled": bool(PREPROCESS_CONFIG.get("enable", True)),
                    "preprocessing_steps": [k for k, v in PREPROCESS_CONFIG.items() if v is True and k != "enable"],
                    "ner_enabled": self.ner_extractor is not None,
                    "cross_platform_mode": True  # Added to indicate cross-platform compatibility
                },
                
                # Sanitize preprocessing metrics to ensure JSON compatibility
                "preprocessing_metrics": self._sanitize_for_json(preprocess_metrics)
            }
            
            metadata_path = OUTPUT_DIR / "metadata" / f"{doc_name}_metadata.json"
            
            # Double-check sanitization before saving
            metadata_sanitized = self._sanitize_for_json(metadata)
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_sanitized, f, indent=2, ensure_ascii=False)
            logger.info(f"  ✓ Metadata saved")
            
            # Calculate final statistics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            avg_time_per_page = duration / len(pages_to_process) if pages_to_process else 0
            
            # ============================================================
            # FINAL SUMMARY
            # ============================================================
            logger.info(f"\n{'='*80}")
            logger.info(f"PROCESSING COMPLETE ✓")
            logger.info(f"{'='*80}")
            logger.info(f"Document: {doc_name}")
            logger.info(f"Pages processed: {len(pages_to_process)}")
            logger.info(f"Success rate: {successful_pages}/{len(pages_to_process)} pages")
            logger.info(f"Average confidence: {quality_report['avg_confidence']:.2%}")
            logger.info(f"Overall quality: {quality_report['overall_quality'].upper()}")
            logger.info(f"Words extracted: {quality_report['total_words_extracted']:,}")
            logger.info(f"Characters extracted: {quality_report['total_characters']:,}")
            
            if quality_report['problem_pages']:
                logger.warning(f"Problem pages: {quality_report['problem_pages']}")
            else:
                logger.info(f"Problem pages: None")
            
            logger.info(f"\nProcessing time:")
            logger.info(f"  Total: {duration:.1f}s")
            logger.info(f"  Per page: {avg_time_per_page:.1f}s")
            
            logger.info(f"\nOutput files:")
            logger.info(f"  📄 Text: {text_output_path}")
            logger.info(f"  📊 Metadata: {metadata_path}")
            logger.info(f"  📈 Quality report: {quality_report_path}")
            logger.info(f"  🔤 Word data: {words_data_path}")
            
            if structure_json_path:
                logger.info(f"  🏗️  Structure: {structure_json_path}")
                logger.info(f"  📋 Structure (text): {structure_txt_path}")
                logger.info(f"  📑 Table of Contents: {toc_path}")
            
            if entities_json_path:
                logger.info(f"  🏷️  Entities: {entities_json_path}")
                logger.info(f"  📝 Entities (report): {entities_txt_path}")
            
            logger.info(f"{'='*80}\n")
            
            return {
                "success": True,
                "document_name": doc_name,
                "pages_processed": len(pages_to_process),
                "pages_successful": successful_pages,
                "quality_report": quality_report,
                "output_files": {
                    "text": str(text_output_path),
                    "metadata": str(metadata_path),
                    "quality_report": str(quality_report_path),
                    "words_data": str(words_data_path),
                    "structure_json": str(structure_json_path) if structure_json_path else None,
                    "structure_txt": str(structure_txt_path) if structure_txt_path else None,
                    "toc": str(toc_path) if toc_path else None,
                    "entities_json": str(entities_json_path) if entities_json_path else None,
                    "entities_txt": str(entities_txt_path) if entities_txt_path else None
                },
                "processing_time": duration,
                "avg_time_per_page": avg_time_per_page
            }
            
        except Exception as e:
            logger.error(f"\n{'='*80}")
            logger.error(f"PROCESSING FAILED ✗")
            logger.error(f"{'='*80}")
            logger.error(f"Document: {pdf_path.name}")
            logger.error(f"Error: {str(e)}")
            logger.error(f"{'='*80}\n")
            
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            
            return {
                "success": False,
                "document_name": pdf_path.stem,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def process_batch(self, pdf_paths: List[Path], max_pages: int = None) -> List[Dict]:
        """
        Process multiple PDFs in batch
        
        Args:
            pdf_paths: List of PDF file paths
            max_pages: Maximum pages per document (None = all)
            
        Returns:
            List of processing results
        """
        results = []
        
        logger.info(f"\n{'#'*80}")
        logger.info(f"BATCH PROCESSING: {len(pdf_paths)} documents")
        logger.info(f"{'#'*80}\n")
        
        for idx, pdf_path in enumerate(pdf_paths, 1):
            logger.info(f"\n>>> Document {idx}/{len(pdf_paths)}")
            result = self.process_document(pdf_path, max_pages=max_pages)
            results.append(result)
        
        return results


def print_batch_summary(results: List[Dict]):
    """Print summary of batch processing"""
    logger.info(f"\n{'#'*80}")
    logger.info(f"BATCH PROCESSING SUMMARY")
    logger.info(f"{'#'*80}")
    
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    
    logger.info(f"Total documents: {len(results)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    
    if successful:
        total_pages = sum(r.get('pages_processed', 0) for r in successful)
        total_time = sum(r.get('processing_time', 0) for r in successful)
        avg_confidence = sum(r['quality_report']['avg_confidence'] for r in successful) / len(successful)
        
        logger.info(f"\nProcessing statistics:")
        logger.info(f"  Total pages processed: {total_pages}")
        logger.info(f"  Total processing time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        logger.info(f"  Average confidence: {avg_confidence:.2%}")
        logger.info(f"  Average speed: {total_time/total_pages:.1f}s per page")
        
        logger.info(f"\nSuccessful documents:")
        for result in successful:
            quality = result['quality_report']['overall_quality']
            confidence = result['quality_report']['avg_confidence']
            pages = result['pages_processed']
            
            status_icon = "✓"
            if quality == "poor":
                status_icon = "⚠"
            elif quality == "excellent":
                status_icon = "⭐"
            
            logger.info(f"  {status_icon} {result['document_name']}: "
                       f"{quality.upper()} ({confidence:.1%}) - {pages} pages")
    
    if failed:
        logger.error(f"\nFailed documents:")
        for result in failed:
            logger.error(f"  ✗ {result['document_name']}: {result.get('error', 'Unknown error')}")
    
    logger.info(f"\n{'#'*80}\n")


def main():
    """Main entry point - Process all PDFs in input directory"""
    
    print(f"\n{'='*80}")
    print(f"OCR DOCUMENT INTELLIGENCE SYSTEM")
    print(f"Cross-Platform Edition (Windows, Linux, macOS)")
    print(f"{'='*80}\n")
    
    # Check if input directory has PDFs
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"⚠ No PDF files found in input directory")
        logger.info(f"\nPlease place your PDF files in:")
        logger.info(f"  {INPUT_DIR.absolute()}")
        logger.info(f"\nCurrent directory structure:")
        logger.info(f"  input_pdfs/    ← Place PDFs here")
        logger.info(f"  output/        ← Results will be saved here")
        logger.info(f"    ├── text/")
        logger.info(f"    ├── metadata/")
        logger.info(f"    ├── quality_reports/")
        logger.info(f"    ├── structure/      ← Document hierarchy")
        logger.info(f"    └── entities/       ← Named entities")
        logger.info(f"  temp/          ← Temporary processing files")
        logger.info(f"  logs/          ← Processing logs")
        return
    
    # Display found PDFs
    logger.info(f"Found {len(pdf_files)} PDF file(s):")
    for pdf_file in pdf_files:
        size_mb = pdf_file.stat().st_size / 1024 / 1024
        logger.info(f"  • {pdf_file.name} ({size_mb:.2f} MB)")
    
    # Ask for confirmation if in interactive mode
    if sys.stdin.isatty():
        logger.info(f"\nPress Enter to start processing (or Ctrl+C to cancel)...")
        try:
            input()
        except KeyboardInterrupt:
            logger.info("\nProcessing cancelled by user")
            return
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Process all PDFs
    # Set max_pages=5 for testing, max_pages=None for full processing
    results = processor.process_batch(pdf_files, max_pages=None)
    
    # Print summary
    print_batch_summary(results)
    
    # Final message
    logger.info(f"All processing complete!")
    logger.info(f"\nCheck results in:")
    logger.info(f"  {OUTPUT_DIR.absolute()}")
    logger.info(f"\nFeatures included:")
    logger.info(f"  ✓ High-quality OCR with preprocessing")
    logger.info(f"  ✓ Document structure analysis (rule-based)")
    logger.info(f"  ✓ Hierarchical section detection")
    logger.info(f"  ✓ Named entity recognition")
    logger.info(f"  ✓ Language detection (offline)")
    logger.info(f"  ✓ Quality assessment & reporting")
    logger.info(f"  ✓ Cross-platform compatibility (Windows, Linux, macOS)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nProcessing interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n\nFatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)