"""
Sound Detection with YAMNet (Lightweight Alternative)
No manual download needed - model auto-downloads (only 5MB)
"""

import sys
from pathlib import Path

# Add detector to path
sys.path.append(str(Path(__file__).parent))

from detector import AudioPreprocessor, ResultProcessor
from detector.yamnet_detector import YAMNetDetector
from config import (
    TARGET_SOUNDS, CONFIDENCE_THRESHOLDS,
    AUDIO_CONFIG, OUTPUT_CONFIG
)

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_sounds_yamnet(audio_path, output_dir="output"):
    """
    Detect sounds using YAMNet model
    
    Args:
        audio_path (str): Path to audio file
        output_dir (str): Output directory
    """
    
    print("=" * 60)
    print("Sound Detection with YAMNet")
    print("=" * 60)
    print(f"Audio: {audio_path}")
    print("=" * 60)
    print()
    
    # Initialize components
    logger.info("Initializing detection pipeline...")
    
    preprocessor = AudioPreprocessor(
        target_sr=AUDIO_CONFIG['sample_rate'],
        window_size=AUDIO_CONFIG['window_size'],
        overlap=AUDIO_CONFIG['overlap']
    )
    
    detector = YAMNetDetector()  # Auto-downloads model
    
    processor = ResultProcessor(
        target_sounds=TARGET_SOUNDS,
        confidence_thresholds=CONFIDENCE_THRESHOLDS,
        min_duration=AUDIO_CONFIG['min_event_duration'],
        max_duration=AUDIO_CONFIG['max_event_duration']
    )
    
    # Process audio
    logger.info("\n[1/4] Preprocessing audio...")
    preprocessed = preprocessor.preprocess(audio_path)
    
    logger.info("\n[2/4] Running YAMNet detection...")
    raw_detections = detector.predict_audio(
        audio=preprocessed['audio'],
        sample_rate=preprocessed['sample_rate'],
        windows_info=preprocessed['windows']
    )
    
    logger.info("\n[3/4] Post-processing results...")
    filtered_detections = processor.process_results(raw_detections)
    
    logger.info("\n[4/4] Saving results...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    (output_path / "reports").mkdir(exist_ok=True)
    (output_path / "segments").mkdir(exist_ok=True)
    
    # Create summary
    summary = processor.create_summary(filtered_detections)
    summary['file'] = Path(audio_path).name
    summary['duration'] = preprocessed['duration']
    
    # Save reports
    base_name = Path(audio_path).stem
    
    json_path = output_path / "reports" / f"{base_name}_detections.json"
    processor.save_json(filtered_detections, summary, Path(audio_path).name, json_path)
    
    csv_path = output_path / "reports" / f"{base_name}_detections.csv"
    processor.save_csv(filtered_detections, csv_path)
    
    report_path = output_path / "reports" / f"{base_name}_report.txt"
    processor.create_report(summary, report_path)
    
    # Save segments
    if filtered_detections and OUTPUT_CONFIG['save_segments']:
        logger.info(f"Saving {len(filtered_detections)} audio segments...")
        
        for det in filtered_detections:
            segment = preprocessor.extract_segment(
                audio=preprocessed['audio'],
                sr=preprocessed['sample_rate'],
                start_time=det['start_time'],
                end_time=det['end_time'],
                padding=OUTPUT_CONFIG['segment_padding']
            )
            
            label_clean = det['label'].replace(' ', '_').replace('/', '_')
            segment_name = f"{base_name}_{det['event_id']:03d}_{label_clean}_{det['start_time']:.1f}s.wav"
            segment_path = output_path / "segments" / segment_name
            
            preprocessor.save_segment(segment, preprocessed['sample_rate'], segment_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Detection Complete!")
    print("=" * 60)
    print(f"Total events detected: {len(filtered_detections)}")
    
    if filtered_detections:
        print(f"\nBy category:")
        for category, count in summary['by_category'].items():
            print(f"  {category}: {count}")
        
        print(f"\nTop detections:")
        for det in filtered_detections[:5]:
            print(f"  - {det['label']} at {det['start_time']:.1f}s (confidence: {det['confidence']:.2%})")
    
    print(f"\nResults saved to: {output_path}/reports/")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect sounds using YAMNet')
    parser.add_argument('--input', '-i', required=True, help='Input audio file')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    
    args = parser.parse_args()
    
    detect_sounds_yamnet(args.input, args.output)
