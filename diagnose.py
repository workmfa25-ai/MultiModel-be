# """
# Quick Diagnostic - See what YAMNet actually detects
# """

# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).parent))

# from detector.yamnet_detector import YAMNetDetector
# from detector.preprocessing import AudioPreprocessor
# from config import AUDIO_CONFIG

# import argparse

# def diagnose(audio_path):
#     print("=" * 70)
#     print("YAMNet Diagnostic - What's Actually Being Detected")
#     print("=" * 70)
#     print(f"Audio: {audio_path}")
#     print("=" * 70)
#     print()
    
#     # Initialize
#     preprocessor = AudioPreprocessor(
#         target_sr=AUDIO_CONFIG['sample_rate'],
#         window_size=AUDIO_CONFIG['window_size'],
#         overlap=AUDIO_CONFIG['overlap']
#     )
    
#     detector = YAMNetDetector()
    
#     # Preprocess
#     print("Loading audio...")
#     preprocessed = preprocessor.preprocess(audio_path)
#     print(f"Duration: {preprocessed['duration']:.2f}s")
#     print(f"Windows: {len(preprocessed['windows'])}")
#     print()
    
#     # Detect
#     print("Running YAMNet detection...")
#     print()
    
#     all_detections = []
    
#     for idx, (window, start_time, end_time) in enumerate(preprocessed['windows']):
#         predictions = detector.predict_window(window, top_k=20)
        
#         print(f"Window {idx+1} ({start_time:.1f}s - {end_time:.1f}s):")
#         print("-" * 70)
        
#         for label, confidence in predictions[:10]:  # Show top 10
#             if confidence >= 0.05:  # Very low threshold
#                 all_detections.append((label, confidence, start_time))
#                 print(f"  {confidence:5.1%}  {label}")
        
#         print()
    
#     # Summary
#     print("=" * 70)
#     print("SUMMARY - Top Detected Sounds Across Entire Audio")
#     print("=" * 70)
    
#     # Group by label
#     label_scores = {}
#     for label, conf, time in all_detections:
#         if label not in label_scores:
#             label_scores[label] = []
#         label_scores[label].append(conf)
    
#     # Calculate average confidence
#     label_avg = {}
#     for label, scores in label_scores.items():
#         label_avg[label] = sum(scores) / len(scores)
    
#     # Sort by average confidence
#     sorted_labels = sorted(label_avg.items(), key=lambda x: x[1], reverse=True)
    
#     print(f"\nTop 20 sounds detected:")
#     print("-" * 70)
#     for i, (label, avg_conf) in enumerate(sorted_labels[:20], 1):
#         count = len(label_scores[label])
#         print(f"{i:2d}. {avg_conf:5.1%}  {label:40s} ({count} times)")
    
#     print()
#     print("=" * 70)
#     print("TIP: If you want to detect these sounds, add them to")
#     print("     config/sound_classes.py in TARGET_SOUNDS")
#     print("=" * 70)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Diagnose what YAMNet detects')
#     parser.add_argument('--input', '-i', required=True, help='Input audio file')
    
#     args = parser.parse_args()
    
#     diagnose(args.input)


import sys
import shutil
import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from utility import extract_mp3_metadata

# Local imports
sys.path.append(str(Path(__file__).parent))

from detector.yamnet_detector import YAMNetDetector
from detector.preprocessing import AudioPreprocessor
from config import AUDIO_CONFIG

# --------------------------------------------------
# App setup
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yamnet-diagnostic")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

# Lazy init
preprocessor = AudioPreprocessor(
    target_sr=AUDIO_CONFIG["sample_rate"],
    window_size=AUDIO_CONFIG["window_size"],
    overlap=AUDIO_CONFIG["overlap"]
)
yamnet = YAMNetDetector()

# --------------------------------------------------
# Endpoint
# --------------------------------------------------
@app.post("/diagnose_yamnet")
async def diagnose_yamnet(audio_file: UploadFile = File(...)):
    try:
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="Empty filename")

        # Save upload
        audio_path = UPLOAD_DIR / audio_file.filename
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio_file.file, f)

        logger.info(f"Diagnosing audio: {audio_path}")

        # Preprocess
        data = preprocessor.preprocess(str(audio_path))
        metadata = extract_mp3_metadata(audio_path)
        all_detections = []
        window_outputs = []

        # Detect per window
        for idx, (window, start, end) in enumerate(data["windows"]):
            preds = yamnet.predict_window(window, top_k=20)

            filtered = [
                {"label": label, "confidence": float(conf)}
                for label, conf in preds
                if conf >= 0.05
            ]

            for p in filtered:
                all_detections.append((p["label"], p["confidence"]))

            window_outputs.append({
                "window_index": idx + 1,
                "start_time": round(start, 2),
                "end_time": round(end, 2),
                "predictions": filtered
            })

        # Aggregate summary
        label_map = {}
        for label, conf in all_detections:
            label_map.setdefault(label, []).append(conf)

        summary = [
            {
                "label": label,
                "avg_confidence": round(sum(scores) / len(scores), 3),
                "count": len(scores)
            }
            for label, scores in label_map.items()
        ]

        summary.sort(key=lambda x: x["avg_confidence"], reverse=True)

        return {
            "engine": "yamnet-diagnostic",
            "file": audio_file.filename,
            "metadata": metadata,
            "duration": round(data["duration"], 2),
            "sample_rate": data["sample_rate"],
            "windows": len(data["windows"]),
            "window_detections": window_outputs,
            "summary": summary[:20]
        }

    except Exception as e:
        logger.exception("Diagnostic failed")
        raise HTTPException(status_code=500, detail=str(e))
