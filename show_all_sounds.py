"""
Show ALL sounds detected by YAMNet (not filtered by target sounds)
Useful to see what's actually in your audio files
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from detector import AudioPreprocessor
from detector.yamnet_detector import YAMNetDetector
from config import AUDIO_CONFIG

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def show_all_detections(audio_path, top_k=20, min_confidence=0.1):
    """Show all sounds detected in audio file"""
    
    print("=" * 70)
    print("YAMNet - Show ALL Detected Sounds")
    print("=" * 70)
    print(f"Audio: {audio_path}")
    print(f"Showing top {top_k} sounds per window (confidence > {min_confidence})")
    print("=" * 70)
    print()
    
    # Initialize
    preprocessor = AudioPreprocessor(
        target_sr=AUDIO_CONFIG['sample_rate'],
        window_size=AUDIO_CONFIG['window_size'],
        overlap=AUDIO_CONFIG['overlap']
    )
    
    detector = YAMNetDetector()
    
    # Preprocess
    print("Loading and preprocessing audio...")
    preprocessed = preprocessor.preprocess(audio_path)
    print(f"Duration: {preprocessed['duration']:.1f}s")
    print(f"Windows: {len(preprocessed['windows'])}")
    print()
    
    # Detect
    print("Detecting sounds...")
    print()
    
    all_sounds = {}
    
    for idx, (window, start_time, end_time) in enumerate(preprocessed['windows']):
        predictions = detector.predict_window(window, top_k=top_k)
        
        print(f"Window {idx+1} ({start_time:.1f}s - {end_time:.1f}s):")
        print("-" * 70)
        
        for label, confidence in predictions:
            if confidence >= min_confidence:
                print(f"  {confidence:5.1%}  {label}")
                
                # Track overall
                if label not in all_sounds:
                    all_sounds[label] = []
                all_sounds[label].append({
                    'time': start_time,
                    'confidence': confidence
                })
        
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY - All Detected Sounds")
    print("=" * 70)
    print(f"Total unique sounds detected: {len(all_sounds)}")
    print()
    
    # Sort by frequency
    sorted_sounds = sorted(all_sounds.items(), key=lambda x: len(x[1]), reverse=True)
    
    print("Most frequent sounds:")
    print("-" * 70)
    for label, occurrences in sorted_sounds[:30]:
        count = len(occurrences)
        avg_conf = sum(o['confidence'] for o in occurrences) / count
        print(f"  {count:3d}x  {avg_conf:5.1%}  {label}")
    
    print()
    print("=" * 70)
    print(f"Full results: {len(sorted_sounds)} different sounds detected")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Show all sounds detected by YAMNet')
    parser.add_argument('--input', '-i', required=True, help='Input audio file')
    parser.add_argument('--top-k', type=int, default=20, help='Top K predictions per window')
    parser.add_argument('--min-confidence', type=float, default=0.1, help='Minimum confidence')
    
    args = parser.parse_args()
    
    show_all_detections(args.input, args.top_k, args.min_confidence)
