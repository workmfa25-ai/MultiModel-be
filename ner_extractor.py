"""
Named Entity Recognition & Metadata Extraction
Extracts entities, detects languages
CROSS-PLATFORM COMPATIBLE (Windows, Linux, macOS)
GEO-TAGGING DISABLED (requires internet, optional feature)
"""
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
import json
import re
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

# Import NLP libraries
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available. Install with: pip install spacy")

try:
    import langdetect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.warning("langdetect not available. Install with: pip install langdetect")

# CROSS-PLATFORM FIX: Make geo-tagging completely optional
# geopy and pycountry are kept optional - if not installed, geo-tagging is disabled
GEOCODING_AVAILABLE = False
try:
    from geopy.geocoders import Nominatim
    import pycountry
    GEOCODING_AVAILABLE = True
    logger.info("Geo-tagging available (requires internet)")
except ImportError:
    logger.info("Geo-tagging disabled (geopy/pycountry not installed - working offline)")


class NERExtractor:
    """Extract named entities and metadata from documents"""
    
    def __init__(self, model_name: str = "en_core_web_sm", enable_geocoding: bool = False):
        """
        Initialize NER extractor
        
        Args:
            model_name: spaCy model to use
                - en_core_web_sm: Small, fast (13MB)
                - en_core_web_md: Medium (43MB)
                - en_core_web_lg: Large (741MB)
                - en_core_web_trf: Transformer (438MB, best accuracy)
            enable_geocoding: Enable geo-tagging (requires internet and geopy/pycountry)
        """
        if not SPACY_AVAILABLE:
            raise ImportError("spaCy is required. Install with: pip install spacy")
        
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"✓ Loaded spaCy model: {model_name}")
        except OSError:
            logger.error(f"Model '{model_name}' not found. Downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name)
        
        # CROSS-PLATFORM FIX: Geo-coding is optional and disabled by default
        self.geocoder = None
        if enable_geocoding and GEOCODING_AVAILABLE:
            try:
                self.geocoder = Nominatim(user_agent="document_intelligence")
                logger.info("✓ Geocoder initialized (requires internet)")
            except Exception as e:
                logger.warning(f"Geocoder initialization failed: {e}")
        else:
            if enable_geocoding and not GEOCODING_AVAILABLE:
                logger.warning("Geo-tagging requested but geopy/pycountry not available")
            logger.info("✓ Working in offline mode (geo-tagging disabled)")
        
        # Custom entity patterns for domain-specific entities
        self._add_custom_patterns()
        
        # Entity normalizer cache
        self._entity_cache = {}
    
    def _add_custom_patterns(self):
        """Add custom entity patterns for domain-specific entities"""
        # Add EntityRuler for custom patterns
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            
            patterns = []
            
            # Military units patterns
            military_patterns = [
                {"label": "MILITARY_UNIT", "pattern": [{"LOWER": {"REGEX": r"\d+(st|nd|rd|th)"}, "OP": "?"}, {"LOWER": "fleet"}]},
                {"label": "MILITARY_UNIT", "pattern": [{"LOWER": {"REGEX": r"\d+(st|nd|rd|th)"}}, {"LOWER": "division"}]},
                {"label": "MILITARY_UNIT", "pattern": [{"LOWER": {"REGEX": r"\d+(st|nd|rd|th)"}}, {"LOWER": "brigade"}]},
                {"label": "MILITARY_UNIT", "pattern": [{"LOWER": {"REGEX": r"\d+(st|nd|rd|th)"}}, {"LOWER": "battalion"}]},
            ]
            patterns.extend(military_patterns)
            
            # Treaty/Agreement patterns
            treaty_patterns = [
                {"label": "TREATY", "pattern": [{"TEXT": {"REGEX": r".*Treaty.*"}}]},
                {"label": "TREATY", "pattern": [{"TEXT": {"REGEX": r".*Agreement.*"}}]},
                {"label": "TREATY", "pattern": [{"TEXT": {"REGEX": r".*Convention.*"}}]},
                {"label": "TREATY", "pattern": [{"TEXT": {"REGEX": r".*Protocol.*"}}]},
            ]
            patterns.extend(treaty_patterns)
            
            ruler.add_patterns(patterns)
            logger.info(f"✓ Added {len(patterns)} custom entity patterns")
    
    def detect_language(self, text: str) -> Optional[Dict]:
        """
        Detect language of text (OFFLINE - no internet required)
        
        Returns:
            Dict with language code and confidence
        """
        if not LANGDETECT_AVAILABLE or not text or len(text.strip()) < 50:
            return None
        
        try:
            # langdetect returns language code (works offline)
            lang_code = langdetect.detect(text)
            
            # Get probabilities for all languages
            lang_probs = langdetect.detect_langs(text)
            confidence = next((p.prob for p in lang_probs if p.lang == lang_code), 0.0)
            
            return {
                "language": lang_code,
                "confidence": confidence,
                "full_name": self._get_language_name(lang_code)
            }
        except Exception as e:
            logger.debug(f"Language detection failed: {e}")
            return None
    
    def _get_language_name(self, code: str) -> str:
        """Get full language name from ISO 639-1 code"""
        lang_names = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ar': 'Arabic',
            'ru': 'Russian',
            'pt': 'Portuguese',
            'it': 'Italian'
        }
        return lang_names.get(code, code.upper())
    
    def extract_entities(self, text: str, min_confidence: float = 0.5) -> Dict:
        """
        Extract named entities from text (OFFLINE - no internet required)
        
        Args:
            text: Input text
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary with extracted entities by type
        """
        if not text or len(text.strip()) < 10:
            return {}
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract entities by type
        entities = defaultdict(list)
        
        for ent in doc.ents:
            entity_info = {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            }
            
            # Normalize entity text
            normalized = self._normalize_entity(ent.text, ent.label_)
            if normalized:
                entity_info["normalized"] = normalized
            
            entities[ent.label_].append(entity_info)
        
        return dict(entities)
    
    def _normalize_entity(self, text: str, label: str) -> Optional[str]:
        """Normalize entity text (remove duplicates, standardize)"""
        # Cache check
        cache_key = f"{label}:{text}"
        if cache_key in self._entity_cache:
            return self._entity_cache[cache_key]
        
        normalized = text.strip()
        
        # Organization normalization
        if label == "ORG":
            # Remove common suffixes
            normalized = re.sub(r'\s+(Inc\.|Corp\.|Ltd\.|LLC)$', '', normalized, flags=re.IGNORECASE)
        
        # Person normalization
        elif label == "PERSON":
            # Title case
            normalized = normalized.title()
        
        # Location normalization
        elif label in ["GPE", "LOC"]:
            # Standardize common location names
            normalized = normalized.strip()
        
        # Cache result
        self._entity_cache[cache_key] = normalized
        return normalized
    
    def extract_dates(self, text: str) -> List[Dict]:
        """Extract and normalize dates from text"""
        doc = self.nlp(text)
        
        dates = []
        for ent in doc.ents:
            if ent.label_ == "DATE":
                date_info = {
                    "text": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char
                }
                
                # Try to parse to ISO format
                normalized = self._normalize_date(ent.text)
                if normalized:
                    date_info["normalized"] = normalized
                
                dates.append(date_info)
        
        return dates
    
    def _normalize_date(self, date_text: str) -> Optional[str]:
        """Attempt to normalize date to ISO format"""
        try:
            from dateutil import parser
            parsed = parser.parse(date_text, fuzzy=True)
            return parsed.strftime("%Y-%m-%d")
        except:
            return None
    
    def geo_tag_location(self, location_text: str) -> Optional[Dict]:
        """
        Geo-tag a location (resolve to coordinates and country)
        REQUIRES INTERNET - disabled in offline mode
        
        Returns:
            Dict with coordinates, country, region (or None if disabled)
        """
        if not self.geocoder or not location_text:
            return None
        
        try:
            # Geocode location (requires internet)
            location = self.geocoder.geocode(location_text, timeout=5)
            
            if not location:
                return None
            
            # Get country info
            country_code = None
            country_name = None
            
            if hasattr(location, 'raw') and 'address' in location.raw:
                country_code = location.raw['address'].get('country_code', '').upper()
                country_name = location.raw['address'].get('country', '')
            
            # Use pycountry for standardization
            if country_code and GEOCODING_AVAILABLE:
                try:
                    country = pycountry.countries.get(alpha_2=country_code)
                    if country:
                        country_name = country.name
                        country_code = country.alpha_2
                except:
                    pass
            
            return {
                "location": location_text,
                "latitude": location.latitude,
                "longitude": location.longitude,
                "country_code": country_code,
                "country_name": country_name,
                "display_name": location.address if hasattr(location, 'address') else location_text
            }
        
        except Exception as e:
            logger.debug(f"Geocoding failed for '{location_text}': {e}")
            return None
    
    def aggregate_entities(self, entities: Dict) -> Dict:
        """
        Aggregate and count entities
        
        Returns:
            Dict with entity counts and top entities
        """
        aggregated = {}
        
        for entity_type, entity_list in entities.items():
            # Count occurrences
            text_counter = Counter([e.get('normalized', e['text']) for e in entity_list])
            
            # Get top entities
            top_entities = [
                {
                    "text": text,
                    "count": count,
                    "normalized": text
                }
                for text, count in text_counter.most_common(20)
            ]
            
            aggregated[entity_type] = {
                "total_count": len(entity_list),
                "unique_count": len(text_counter),
                "top_entities": top_entities
            }
        
        return aggregated
    
    def extract_document_entities(
        self,
        structure_analysis: Dict,
        full_text: str
    ) -> Dict:
        """
        Extract entities from entire document using structure
        
        Args:
            structure_analysis: Document structure from layout analyzer
            full_text: Complete document text
            
        Returns:
            Comprehensive entity extraction results
        """
        logger.info("Extracting entities from document...")
        
        # Detect document language (offline)
        doc_language = self.detect_language(full_text[:5000])  # Use first 5000 chars
        
        # Extract entities from full text (offline)
        all_entities = self.extract_entities(full_text)
        
        # Extract dates (offline)
        dates = self.extract_dates(full_text)
        if dates:
            all_entities['DATE'] = dates
        
        # Aggregate entities
        entity_summary = self.aggregate_entities(all_entities)
        
        # Geo-tag locations (requires internet - may be disabled)
        geo_tags = self._geo_tag_all_locations(all_entities)
        
        # Extract section-level entities if structure available
        section_entities = {}
        if structure_analysis and 'hierarchy' in structure_analysis:
            section_entities = self._extract_section_entities(
                structure_analysis['hierarchy']
            )
        
        result = {
            "document_language": doc_language,
            "entity_summary": entity_summary,
            "all_entities": all_entities,
            "geo_tags": geo_tags,
            "section_entities": section_entities,
            "statistics": {
                "total_entities": sum(
                    summary['total_count'] 
                    for summary in entity_summary.values()
                ),
                "unique_entities": sum(
                    summary['unique_count'] 
                    for summary in entity_summary.values()
                ),
                "entity_types": list(entity_summary.keys())
            }
        }
        
        logger.info(f"✓ Extracted {result['statistics']['total_entities']} entities")
        logger.info(f"  Types: {', '.join(result['statistics']['entity_types'])}")
        
        return result
    
    def _geo_tag_all_locations(self, entities: Dict) -> Dict:
        """
        Geo-tag all location entities
        REQUIRES INTERNET - returns empty if disabled
        """
        geo_tags = {
            "locations": [],
            "countries": [],
            "regions": []
        }
        
        # Skip if geocoder not available
        if not self.geocoder:
            logger.debug("Geo-tagging skipped (disabled or offline mode)")
            return geo_tags
        
        # Get all location entities
        locations = []
        for label in ['GPE', 'LOC', 'FAC']:
            if label in entities:
                locations.extend(entities[label])
        
        # Geo-tag unique locations (limit to avoid rate limiting)
        unique_locations = list(set([
            loc.get('normalized', loc['text']) 
            for loc in locations
        ]))[:50]  # Limit to 50 to avoid excessive API calls
        
        countries_set = set()
        regions_set = set()
        
        for location_text in unique_locations:
            geo_info = self.geo_tag_location(location_text)
            if geo_info:
                geo_tags["locations"].append(geo_info)
                if geo_info.get('country_code'):
                    countries_set.add(geo_info['country_code'])
                if geo_info.get('country_name'):
                    regions_set.add(geo_info['country_name'])
        
        # Convert sets to lists for JSON serialization
        geo_tags["countries"] = list(countries_set)
        geo_tags["regions"] = list(regions_set)
        
        return geo_tags
    
    def _extract_section_entities(self, hierarchy: Dict) -> Dict:
        """Extract entities per section"""
        section_entities = {}
        
        def traverse(node, section_path=""):
            if node.get('type') in ['section', 'subsection']:
                section_id = node.get('element_id', '')
                section_title = node.get('text', '')
                section_number = node.get('number', '')
                
                # Create section identifier
                if section_number:
                    current_path = f"{section_path}/{section_number}"
                else:
                    current_path = f"{section_path}/{section_title[:30]}"
                
                # Extract entities from section text
                if section_title:
                    section_ents = self.extract_entities(section_title)
                    if section_ents:
                        section_entities[current_path] = {
                            "title": section_title,
                            "number": section_number,
                            "entities": section_ents
                        }
                
                # Recurse into children
                for child in node.get('children', []):
                    traverse(child, current_path)
            
            elif node.get('type') == 'paragraph':
                # Could extract from paragraphs too, but might be too granular
                pass
            
            else:
                # Recurse for document root
                for child in node.get('children', []):
                    traverse(child, section_path)
        
        traverse(hierarchy)
        return section_entities


def save_entities_report(entities: Dict, output_path: Path):
    """Save entities in human-readable format"""
    lines = []
    
    lines.append("=" * 80)
    lines.append("ENTITY EXTRACTION REPORT")
    lines.append("=" * 80)
    
    # Document language
    if entities.get('document_language'):
        lang = entities['document_language']
        lines.append(f"\nDocument Language: {lang['full_name']} ({lang['language']}) "
                    f"- Confidence: {lang['confidence']:.2%}")
    
    # Statistics
    stats = entities.get('statistics', {})
    lines.append(f"\nTotal Entities: {stats.get('total_entities', 0)}")
    lines.append(f"Unique Entities: {stats.get('unique_entities', 0)}")
    lines.append(f"Entity Types: {len(stats.get('entity_types', []))}")
    
    # Top entities by type
    lines.append("\n" + "=" * 80)
    lines.append("TOP ENTITIES BY TYPE")
    lines.append("=" * 80)
    
    summary = entities.get('entity_summary', {})
    for entity_type, data in sorted(summary.items()):
        lines.append(f"\n{entity_type} ({data['unique_count']} unique, {data['total_count']} total):")
        for ent in data['top_entities'][:10]:
            lines.append(f"  • {ent['text']} ({ent['count']}x)")
    
    # Geo-tags (may be empty if disabled)
    geo_tags = entities.get('geo_tags', {})
    if geo_tags.get('countries'):
        lines.append("\n" + "=" * 80)
        lines.append("GEOGRAPHIC COVERAGE")
        lines.append("=" * 80)
        lines.append(f"\nCountries: {', '.join(sorted(geo_tags['countries']))}")
        
        if geo_tags.get('locations'):
            lines.append(f"\nGeo-tagged Locations ({len(geo_tags['locations'])}):")
            for loc in geo_tags['locations'][:20]:
                lines.append(f"  • {loc['location']} → {loc.get('country_name', 'Unknown')}")
    
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Entity report saved to: {output_path}")


if __name__ == "__main__":
    # Test NER extractor
    import sys
    
    if len(sys.argv) > 1:
        # Load structure file
        structure_path = Path(sys.argv[1])
        with open(structure_path, 'r') as f:
            structure = json.load(f)
        
        # Load text file
        text_path = structure_path.parent.parent / "text" / f"{structure['document_name']}.txt"
        if text_path.exists():
            full_text = text_path.read_text(encoding='utf-8')
        else:
            print(f"Text file not found: {text_path}")
            sys.exit(1)
        
        # Extract entities (offline mode, no geo-tagging)
        extractor = NERExtractor(model_name="en_core_web_sm", enable_geocoding=False)
        entities = extractor.extract_document_entities(structure, full_text)
        
        # Save results
        output_dir = structure_path.parent.parent / "entities"
        output_dir.mkdir(exist_ok=True)
        
        json_path = output_dir / f"{structure['document_name']}_entities.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(entities, f, indent=2, ensure_ascii=False, default=str)
        
        txt_path = output_dir / f"{structure['document_name']}_entities.txt"
        save_entities_report(entities, txt_path)
        
        print(f"\n✓ Entity extraction complete!")
        print(f"  JSON: {json_path}")
        print(f"  Report: {txt_path}")
    else:
        print("Usage: python ner_extractor.py <structure_json_path>")