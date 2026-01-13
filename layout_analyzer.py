"""
Layout Analysis: Detect document structure and hierarchy
Handles dense analytical reports with sections, subsections, tables, figures
CROSS-PLATFORM COMPATIBLE (Windows, Linux, macOS)
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional
import re
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger(__name__)

# CROSS-PLATFORM FIX: Disable layoutparser (Linux-only dependency)
# layoutparser requires detectron2 which doesn't work on Windows
LAYOUTPARSER_AVAILABLE = False
logger.info("LayoutParser disabled for cross-platform compatibility. Using rule-based analysis.")


@dataclass
class DocumentElement:
    """Represents a structural element in the document"""
    element_id: str
    type: str  # 'title', 'section', 'subsection', 'paragraph', 'table', 'figure', 'footnote', 'list'
    text: str
    page: int
    bbox: Dict  # {'x': int, 'y': int, 'width': int, 'height': int}
    level: int  # Hierarchical depth (0=title, 1=section, 2=subsection, etc.)
    number: str = ""  # Section numbering (e.g., "3.2.1")
    confidence: float = 0.0
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
    
    def to_dict(self):
        return asdict(self)


class LayoutAnalyzer:
    """Analyze document layout and reconstruct hierarchy"""
    
    def __init__(self, use_ml_model: bool = False):
        """
        Initialize layout analyzer
        
        Args:
            use_ml_model: Whether to use ML-based layout detection (NOT AVAILABLE - cross-platform mode)
        """
        # CROSS-PLATFORM FIX: Always use rule-based analysis
        self.use_ml_model = False
        
        if use_ml_model:
            logger.warning(
                "ML-based layout detection is disabled for cross-platform compatibility. "
                "Using rule-based analysis which works on Windows, Linux, and macOS."
            )
        
        # Section number patterns (ordered by priority)
        self.section_patterns = [
            (r'^(\d+\.)+\d+\s+', 'decimal'),      # 1.2.3
            (r'^(\d+\.)+\s+', 'decimal'),          # 1.2.
            (r'^[IVX]+\.\s+', 'roman'),            # I. II. III.
            (r'^[A-Z]\.\s+', 'letter_upper'),      # A. B. C.
            (r'^[a-z]\)\s+', 'letter_lower'),      # a) b) c)
            (r'^\(\d+\)\s+', 'paren_number'),      # (1) (2) (3)
        ]
        
        # Common header/title keywords
        self.title_keywords = [
            'abstract', 'introduction', 'conclusion', 'references', 'bibliography',
            'summary', 'overview', 'background', 'methodology', 'results',
            'discussion', 'acknowledgments', 'appendix', 'executive summary',
            'table of contents', 'preface', 'foreword'
        ]
        
        # Footnote indicators
        self.footnote_patterns = [
            r'^\d+\s+',           # Numbered footnotes
            r'^\*+\s+',           # Asterisk footnotes
            r'^†+\s+',            # Dagger footnotes
        ]
    
    def analyze_page_structure(
        self,
        ocr_result,
        page_num: int,
        page_image_path: Optional[Path] = None
    ) -> List[DocumentElement]:
        """
        Analyze structure of a single page
        
        Args:
            ocr_result: OCR result object with words
            page_num: Page number
            page_image_path: Optional path to page image (ignored in cross-platform mode)
            
        Returns:
            List of DocumentElement objects
        """
        # CROSS-PLATFORM: Only use rule-based analysis
        return self._rule_based_detection(ocr_result, page_num)
    
    def _rule_based_detection(self, ocr_result, page_num: int) -> List[DocumentElement]:
        """Rule-based structure detection from OCR"""
        elements = []
        
        # Group words into lines
        lines = self._group_words_into_lines(ocr_result.words)
        
        # Analyze each line
        for line_idx, line in enumerate(lines):
            element = self._classify_line(line, page_num, line_idx)
            if element:
                elements.append(element)
        
        return elements
    
    def _group_words_into_lines(self, words: List[Dict]) -> List[List[Dict]]:
        """Group words into lines based on y-coordinates"""
        if not words:
            return []
        
        # Sort by y position, then x position
        sorted_words = sorted(words, key=lambda w: (w['bbox']['y'], w['bbox']['x']))
        
        lines = []
        current_line = [sorted_words[0]]
        current_y = sorted_words[0]['bbox']['y']
        
        # Threshold for same line (pixels)
        line_threshold = 15
        
        for word in sorted_words[1:]:
            word_y = word['bbox']['y']
            
            # If word is on same line
            if abs(word_y - current_y) < line_threshold:
                current_line.append(word)
            else:
                # Start new line
                if current_line:
                    lines.append(current_line)
                current_line = [word]
                current_y = word_y
        
        # Add last line
        if current_line:
            lines.append(current_line)
        
        # Sort words within each line by x position
        for line in lines:
            line.sort(key=lambda w: w['bbox']['x'])
        
        return lines
    
    def _classify_line(
        self,
        line_words: List[Dict],
        page_num: int,
        line_idx: int
    ) -> Optional[DocumentElement]:
        """Classify a line as title, section, paragraph, etc."""
        if not line_words:
            return None
        
        # Combine words into text
        text = ' '.join(w['text'] for w in line_words).strip()
        
        if not text:
            return None
        
        avg_confidence = sum(w['confidence'] for w in line_words) / len(line_words)
        
        # Calculate line bbox
        x_min = min(w['bbox']['x'] for w in line_words)
        y_min = min(w['bbox']['y'] for w in line_words)
        x_max = max(w['bbox']['x'] + w['bbox']['width'] for w in line_words)
        y_max = max(w['bbox']['y'] + w['bbox']['height'] for w in line_words)
        
        line_bbox = {
            'x': x_min,
            'y': y_min,
            'width': x_max - x_min,
            'height': y_max - y_min
        }
        
        element_id = f"rule_p{page_num}_l{line_idx}"
        
        # Check for footnote
        if self._is_footnote(text, line_bbox):
            return DocumentElement(
                element_id=element_id,
                type='footnote',
                text=text,
                page=page_num,
                bbox=line_bbox,
                level=999,
                confidence=avg_confidence
            )
        
        # Detect section numbering
        section_info = self._extract_section_info(text)
        
        if section_info:
            level = section_info['level']
            return DocumentElement(
                element_id=element_id,
                type='section' if level == 1 else 'subsection',
                text=text,
                page=page_num,
                bbox=line_bbox,
                level=level,
                number=section_info['number'],
                confidence=avg_confidence
            )
        
        # Check if it's a title/header
        if self._is_likely_header(text, line_bbox):
            return DocumentElement(
                element_id=element_id,
                type='title',
                text=text,
                page=page_num,
                bbox=line_bbox,
                level=0,
                confidence=avg_confidence
            )
        
        # Check if it's a list item
        if self._is_list_item(text):
            return DocumentElement(
                element_id=element_id,
                type='list',
                text=text,
                page=page_num,
                bbox=line_bbox,
                level=999,
                confidence=avg_confidence
            )
        
        # Default: paragraph
        return DocumentElement(
            element_id=element_id,
            type='paragraph',
            text=text,
            page=page_num,
            bbox=line_bbox,
            level=999,  # Will be assigned based on parent section
            confidence=avg_confidence
        )
    
    def _extract_section_info(self, text: str) -> Optional[Dict]:
        """Extract section number and calculate level"""
        for pattern, pattern_type in self.section_patterns:
            match = re.match(pattern, text.strip())
            if match:
                number_str = match.group(0).strip()
                
                # Calculate level based on pattern type
                if pattern_type == 'decimal':
                    # Count dots to determine level
                    level = number_str.rstrip('.').count('.') + 1
                elif pattern_type == 'roman':
                    level = 1
                elif pattern_type == 'letter_upper':
                    level = 2
                elif pattern_type == 'letter_lower':
                    level = 3
                else:
                    level = 2
                
                return {
                    'number': number_str,
                    'level': level,
                    'type': pattern_type
                }
        
        return None
    
    def _is_likely_header(self, text: str, bbox: Dict) -> bool:
        """Determine if text is likely a header/title"""
        text_lower = text.lower().strip()
        
        # Check for title keywords
        if any(keyword in text_lower for keyword in self.title_keywords):
            return True
        
        # Check if all caps (common for headers) and not too long
        if text.isupper() and 3 <= len(text.split()) <= 10:
            return True
        
        # Check if title case and short
        if text.istitle() and len(text.split()) <= 8:
            return True
        
        # Check if ends with colon (often section headers)
        if text.endswith(':') and len(text.split()) <= 8:
            return True
        
        return False
    
    def _is_footnote(self, text: str, bbox: Dict) -> bool:
        """Check if text is a footnote"""
        # Check for footnote patterns at start
        for pattern in self.footnote_patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def _is_list_item(self, text: str) -> bool:
        """Check if text is a list item"""
        list_patterns = [
            r'^•\s+',           # Bullet
            r'^-\s+',           # Dash
            r'^○\s+',           # Circle
            r'^□\s+',           # Square
            r'^\d+\)\s+',       # Numbered with paren
        ]
        
        for pattern in list_patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def build_document_hierarchy(
        self,
        all_elements: List[DocumentElement]
    ) -> Dict:
        """
        Build hierarchical document structure from flat list of elements
        
        Returns:
            Nested dictionary representing document structure
        """
        # Sort elements by page and y-position
        sorted_elements = sorted(
            all_elements,
            key=lambda e: (e.page, e.bbox['y'])
        )
        
        # Build hierarchy
        hierarchy = {
            "type": "document",
            "element_id": "root",
            "children": []
        }
        
        section_stack = [hierarchy]  # Stack to track nested sections
        
        for element in sorted_elements:
            element_dict = {
                "element_id": element.element_id,
                "type": element.type,
                "text": element.text,
                "number": element.number,
                "level": element.level,
                "page": element.page,
                "bbox": element.bbox,
                "confidence": element.confidence,
                "children": []
            }
            
            if element.type in ['title', 'section', 'subsection']:
                # Find appropriate parent in stack
                while len(section_stack) > 1 and section_stack[-1].get('level', 0) >= element.level:
                    section_stack.pop()
                
                # Add to parent
                section_stack[-1]['children'].append(element_dict)
                
                # Push to stack for potential children
                section_stack.append(element_dict)
            
            else:
                # Add to current section (last in stack)
                section_stack[-1]['children'].append(element_dict)
        
        return hierarchy
    
    def analyze_document(
        self,
        ocr_results: List,
        document_name: str,
        image_paths: Optional[List[Path]] = None
    ) -> Dict:
        """
        Analyze entire document structure
        
        Args:
            ocr_results: List of OCR results for each page
            document_name: Name of document
            image_paths: Optional list of paths to page images (ignored in cross-platform mode)
            
        Returns:
            Complete document structure analysis
        """
        logger.info(f"Analyzing structure for: {document_name}")
        
        all_elements = []
        
        # Analyze each page
        for page_num, ocr_result in enumerate(ocr_results, 1):
            if ocr_result:
                page_elements = self.analyze_page_structure(
                    ocr_result,
                    page_num,
                    None  # Don't use image path in cross-platform mode
                )
                all_elements.extend(page_elements)
        
        # Build hierarchy
        hierarchy = self.build_document_hierarchy(all_elements)
        
        # Generate statistics
        type_counts = {}
        for element in all_elements:
            type_counts[element.type] = type_counts.get(element.type, 0) + 1
        
        stats = {
            "total_elements": len(all_elements),
            "type_distribution": type_counts,
            "max_depth": max([e.level for e in all_elements if e.level < 999], default=0),
            "pages_analyzed": len(set(e.page for e in all_elements))
        }
        
        logger.info(f"Structure analysis complete:")
        for elem_type, count in type_counts.items():
            logger.info(f"  {elem_type}: {count}")
        logger.info(f"  Max depth: {stats['max_depth']}")
        
        return {
            "document_name": document_name,
            "hierarchy": hierarchy,
            "elements": [e.to_dict() for e in all_elements],
            "statistics": stats
        }
    
    def extract_table_of_contents(self, hierarchy: Dict) -> List[Dict]:
        """Extract table of contents from hierarchy"""
        toc = []
        
        def traverse(node, depth=0):
            if node.get('type') in ['section', 'subsection', 'title']:
                toc.append({
                    'level': depth,
                    'number': node.get('number', ''),
                    'title': node.get('text', ''),
                    'page': node.get('page', 0),
                    'type': node.get('type')
                })
            
            for child in node.get('children', []):
                traverse(child, depth + 1)
        
        traverse(hierarchy)
        return toc


def save_structure_visualization(structure: Dict, output_path: Path):
    """Save a text visualization of document structure"""
    lines = []
    
    def traverse(node, indent=0):
        if node.get('type') == 'document':
            lines.append("📄 DOCUMENT STRUCTURE\n" + "="*80)
        else:
            prefix = "  " * indent
            icon_map = {
                'title': '📌',
                'section': '📑',
                'subsection': '📝',
                'paragraph': '📄',
                'list': '•',
                'table': '📊',
                'figure': '🖼️',
                'footnote': '📎'
            }
            icon = icon_map.get(node.get('type'), '▪')
            
            number = node.get('number', '')
            text = node.get('text', '')[:100]  # Limit length
            page = node.get('page', '')
            
            line = f"{prefix}{icon} {number} {text}"
            if page:
                line += f" (Page {page})"
            lines.append(line)
        
        for child in node.get('children', []):
            traverse(child, indent + 1)
    
    traverse(structure['hierarchy'])
    
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Structure visualization saved to: {output_path}")


if __name__ == "__main__":
    # Test with command line
    import sys
    if len(sys.argv) > 1:
        from pathlib import Path
        import json
        from ocr_engine import OCRResult
        
        # Load OCR results
        words_json_path = Path(sys.argv[1])
        with open(words_json_path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct OCR results
        ocr_results = []
        for page_data in data['pages']:
            result = OCRResult()
            result.words = page_data['words']
            result.confidence = page_data['confidence']
            result.engine = page_data['engine']
            ocr_results.append(result)
        
        # Analyze
        analyzer = LayoutAnalyzer(use_ml_model=False)
        structure = analyzer.analyze_document(
            ocr_results,
            data['document_name']
        )
        
        # Save
        output_dir = Path("output/structure")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        json_path = output_dir / f"{data['document_name']}_structure.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(structure, f, indent=2, ensure_ascii=False)
        
        txt_path = output_dir / f"{data['document_name']}_structure.txt"
        save_structure_visualization(structure, txt_path)
        
        print(f"\n✓ Structure analysis complete!")
        print(f"  JSON: {json_path}")
        print(f"  Visualization: {txt_path}")
    else:
        print("Usage: python layout_analyzer.py <words_json_path>")