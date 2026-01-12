"""
Post-processing Module
Filters and aggregates detection results
"""

import numpy as np
from pathlib import Path
import logging
from collections import defaultdict
import json
import csv
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResultProcessor:
    """Post-processes detection results"""
    
    def __init__(self, target_sounds, confidence_thresholds, 
                 min_duration=0.1, max_duration=5.0):
        """
        Initialize result processor
        
        Args:
            target_sounds (dict): Dictionary of target sound categories
            confidence_thresholds (dict): Confidence thresholds per category
            min_duration (float): Minimum event duration in seconds
            max_duration (float): Maximum event duration in seconds
        """
        self.target_sounds = target_sounds
        self.confidence_thresholds = confidence_thresholds
        self.min_duration = min_duration
        self.max_duration = max_duration
    
    def get_category(self, label):
        """
        Get category for a label
        
        Args:
            label (str): Sound label
            
        Returns:
            str: Category name or 'other'
        """
        label_lower = label.lower()
        for category, sounds in self.target_sounds.items():
            for sound in sounds:
                if sound.lower() in label_lower or label_lower in sound.lower():
                    return category
        return 'other'
    
    def get_threshold(self, category):
        """Get confidence threshold for category"""
        return self.confidence_thresholds.get(
            category, 
            self.confidence_thresholds.get('default', 0.6)
        )
    
    def filter_detections(self, detections):
        """
        Filter detections by target sounds and confidence
        
        Args:
            detections (list): Raw detections from model
            
        Returns:
            list: Filtered detections
        """
        filtered = []
        
        for detection in detections:
            label = detection['label']
            confidence = detection['confidence']
            category = self.get_category(label)
            
            # Skip if not a target sound
            if category == 'other':
                continue
            
            # Check confidence threshold
            threshold = self.get_threshold(category)
            if confidence < threshold:
                continue
            
            # Add category to detection
            detection['category'] = category
            filtered.append(detection)
        
        logger.info(f"Filtered {len(detections)} -> {len(filtered)} detections")
        return filtered
    
    def merge_overlapping_events(self, detections, iou_threshold=0.3):
        """
        Merge overlapping detections of the same sound
        
        Args:
            detections (list): Filtered detections
            iou_threshold (float): IoU threshold for merging
            
        Returns:
            list: Merged detections
        """
        if not detections:
            return []
        
        # Group by label
        grouped = defaultdict(list)
        for det in detections:
            grouped[det['label']].append(det)
        
        merged = []
        
        for label, dets in grouped.items():
            # Sort by start time
            dets = sorted(dets, key=lambda x: x['start_time'])
            
            current = dets[0].copy()
            
            for next_det in dets[1:]:
                # Calculate overlap
                overlap_start = max(current['start_time'], next_det['start_time'])
                overlap_end = min(current['end_time'], next_det['end_time'])
                overlap = max(0, overlap_end - overlap_start)
                
                # Calculate IoU
                union = (current['end_time'] - current['start_time']) + \
                        (next_det['end_time'] - next_det['start_time']) - overlap
                iou = overlap / union if union > 0 else 0
                
                if iou > iou_threshold:
                    # Merge
                    current['end_time'] = max(current['end_time'], next_det['end_time'])
                    current['confidence'] = max(current['confidence'], next_det['confidence'])
                else:
                    # Save current and start new
                    merged.append(current)
                    current = next_det.copy()
            
            # Add last event
            merged.append(current)
        
        logger.info(f"Merged {len(detections)} -> {len(merged)} events")
        return merged
    
    def remove_short_events(self, detections):
        """Remove events shorter than minimum duration"""
        filtered = []
        for det in detections:
            duration = det['end_time'] - det['start_time']
            if duration >= self.min_duration and duration <= self.max_duration:
                det['duration'] = duration
                filtered.append(det)
        
        return filtered
    
    def process_results(self, detections):
        """
        Complete post-processing pipeline
        
        Args:
            detections (list): Raw detections
            
        Returns:
            list: Processed detections
        """
        # Filter by target sounds and confidence
        filtered = self.filter_detections(detections)
        
        # Merge overlapping events
        merged = self.merge_overlapping_events(filtered)
        
        # Remove short events
        final = self.remove_short_events(merged)
        
        # Add event IDs
        for idx, det in enumerate(final, 1):
            det['event_id'] = idx
        
        # Sort by time
        final = sorted(final, key=lambda x: x['start_time'])
        
        logger.info(f"Final: {len(final)} events")
        return final
    
    def create_summary(self, detections):
        """
        Create summary statistics
        
        Args:
            detections (list): Processed detections
            
        Returns:
            dict: Summary statistics
        """
        summary = {
            'total_events': len(detections),
            'by_category': defaultdict(int),
            'by_label': defaultdict(int),
            'avg_confidence': 0.0,
            'time_range': {'start': None, 'end': None}
        }
        
        if not detections:
            return summary
        
        confidences = []
        
        for det in detections:
            summary['by_category'][det['category']] += 1
            summary['by_label'][det['label']] += 1
            confidences.append(det['confidence'])
        
        summary['avg_confidence'] = np.mean(confidences)
        summary['time_range']['start'] = min(d['start_time'] for d in detections)
        summary['time_range']['end'] = max(d['end_time'] for d in detections)
        
        # Convert defaultdict to dict
        summary['by_category'] = dict(summary['by_category'])
        summary['by_label'] = dict(summary['by_label'])
        
        return summary
    
    def save_json(self, detections, summary, file_name, output_path):
        """
        Save results as JSON
        
        Args:
            detections (list): Processed detections
            summary (dict): Summary statistics
            file_name (str): Original audio file name
            output_path (str): Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        result = {
            'file': file_name,
            'timestamp': datetime.now().isoformat(),
            'summary': summary,
            'detections': detections
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved JSON: {output_path.name}")
    
    def save_csv(self, detections, output_path):
        """
        Save results as CSV
        
        Args:
            detections (list): Processed detections
            output_path (str): Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not detections:
            logger.warning("No detections to save")
            return
        
        # Define CSV columns
        fieldnames = [
            'event_id', 'label', 'category', 'start_time', 'end_time',
            'duration', 'confidence', 'window_center'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for det in detections:
                row = {k: det.get(k, '') for k in fieldnames}
                writer.writerow(row)
        
        logger.info(f"Saved CSV: {output_path.name}")
    
    def create_report(self, summary, output_path):
        """
        Create human-readable text report
        
        Args:
            summary (dict): Summary statistics
            output_path (str): Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("SOUND EVENT DETECTION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total Events Detected: {summary['total_events']}\n")
            f.write(f"Average Confidence: {summary['avg_confidence']:.2%}\n\n")
            
            if summary['time_range']['start'] is not None:
                f.write(f"Time Range: {summary['time_range']['start']:.2f}s - "
                       f"{summary['time_range']['end']:.2f}s\n\n")
            
            f.write("Events by Category:\n")
            f.write("-" * 40 + "\n")
            for category, count in sorted(summary['by_category'].items()):
                f.write(f"  {category.replace('_', ' ').title()}: {count}\n")
            
            f.write("\nEvents by Label:\n")
            f.write("-" * 40 + "\n")
            for label, count in sorted(summary['by_label'].items(), 
                                      key=lambda x: x[1], reverse=True):
                f.write(f"  {label}: {count}\n")
        
        logger.info(f"Saved report: {output_path.name}")
