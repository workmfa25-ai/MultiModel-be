"""
Quality Checker: Assess OCR quality and generate reports
"""
from typing import Dict, List
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class QualityChecker:
    """Assess OCR quality and generate reports"""
    
    def __init__(self, thresholds: Dict):
        self.thresholds = thresholds
    
    def assess_ocr_result(self, ocr_result, page_num: int) -> Dict:
        """Assess quality of OCR result for a single page"""
        confidence = ocr_result.confidence
        
        # Determine quality level
        if confidence >= self.thresholds["excellent"] / 100:
            quality_level = "excellent"
        elif confidence >= self.thresholds["good"] / 100:
            quality_level = "good"
        elif confidence >= self.thresholds["acceptable"] / 100:
            quality_level = "acceptable"
        elif confidence >= self.thresholds["poor"] / 100:
            quality_level = "poor"
        else:
            quality_level = "unreadable"
        
        # Detect issues
        issues = []
        
        if confidence < self.thresholds["acceptable"] / 100:
            issues.append("low_confidence")
        
        if len(ocr_result.words) < 10:
            issues.append("too_few_words")
        
        # Check for OCR errors
        single_char_words = sum(1 for w in ocr_result.words if len(w['text']) == 1)
        if single_char_words > len(ocr_result.words) * 0.3:
            issues.append("excessive_single_characters")
        
        # Check for gibberish
        total_chars = len(ocr_result.text)
        if total_chars > 0:
            alphanum_chars = sum(c.isalnum() or c.isspace() for c in ocr_result.text)
            alphanum_ratio = alphanum_chars / total_chars
            if alphanum_ratio < 0.7:
                issues.append("possible_gibberish")
        
        return {
            "page": page_num,
            "confidence": confidence,
            "quality_level": quality_level,
            "word_count": len(ocr_result.words),
            "char_count": len(ocr_result.text),
            "issues": issues,
            "requires_review": len(issues) > 0 or quality_level in ["poor", "unreadable"],
            "engine_used": ocr_result.engine
        }
    
    def generate_document_report(self, page_reports: List[Dict], document_name: str) -> Dict:
        """Generate overall quality report for entire document"""
        if not page_reports:
            return {"error": "No page reports provided"}
        
        # Calculate averages
        avg_confidence = sum(p["confidence"] for p in page_reports) / len(page_reports)
        total_words = sum(p["word_count"] for p in page_reports)
        total_chars = sum(p["char_count"] for p in page_reports)
        
        # Count pages by quality level
        quality_distribution = {}
        for report in page_reports:
            level = report["quality_level"]
            quality_distribution[level] = quality_distribution.get(level, 0) + 1
        
        # Identify problem pages
        problem_pages = [p["page"] for p in page_reports if p["requires_review"]]
        
        # Overall assessment
        if avg_confidence >= self.thresholds["good"] / 100:
            overall_quality = "good"
            recommendation = "Document processed successfully"
        elif avg_confidence >= self.thresholds["acceptable"] / 100:
            overall_quality = "acceptable"
            recommendation = "Review flagged pages"
        else:
            overall_quality = "poor"
            recommendation = "Consider re-scanning or manual transcription"
        
        return {
            "document_name": document_name,
            "total_pages": len(page_reports),
            "avg_confidence": avg_confidence,
            "overall_quality": overall_quality,
            "quality_distribution": quality_distribution,
            "total_words_extracted": total_words,
            "total_characters": total_chars,
            "problem_pages": problem_pages,
            "problem_page_count": len(problem_pages),
            "recommendation": recommendation,
            "page_details": page_reports
        }
    
    def save_report(self, report: Dict, output_path: Path):
        """Save quality report as JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Quality report saved to {output_path}")
