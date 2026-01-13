
import datetime
import os
import sys
import shutil
import logging
from pathlib import Path
import tempfile
from typing import List

from fastapi import Depends, FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Form
import whisper
# from transformers import Optional

from db import get_db
from models import AudioDiagnosticRun, AudioLabelSummary, AudioWindowDetection, AudioWindowLabel
from utility import extract_mp3_metadata

# Local imports
sys.path.append(str(Path(__file__).parent))

from detector.yamnet_detector import YAMNetDetector
from detector.preprocessing import AudioPreprocessor
from config import AUDIO_CONFIG
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List
from datetime import datetime

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
async def diagnose_yamnet(
    audio_file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    try:
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="Empty filename")

        # Save upload
        audio_path = UPLOAD_DIR / audio_file.filename
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio_file.file, f)

        data = preprocessor.preprocess(str(audio_path))
        metadata = extract_mp3_metadata(audio_path)

        run = AudioDiagnosticRun(
            filename=audio_file.filename,
            duration=round(data["duration"], 2),
            sample_rate=data["sample_rate"],
            windows=len(data["windows"]),
            metadata=metadata
        )
        db.add(run)
        await db.flush()  # get run.id

        all_detections = []
        window_outputs = []

        for idx, (window, start, end) in enumerate(data["windows"]):
            preds = yamnet.predict_window(window, top_k=20)

            filtered = [
                {"label": label, "confidence": float(conf)}
                for label, conf in preds
                if conf >= 0.05
            ]

            window_row = AudioWindowDetection(
                run_id=run.id,
                window_index=idx + 1,
                start_time=round(start, 2),
                end_time=round(end, 2)
            )
            db.add(window_row)
            await db.flush()

            for p in filtered:
                db.add(AudioWindowLabel(
                    window_id=window_row.id,
                    label=p["label"],
                    confidence=p["confidence"]
                ))
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

        summary = []
        for label, scores in label_map.items():
            avg_conf = round(sum(scores) / len(scores), 3)
            count = len(scores)

            summary.append({
                "label": label,
                "confidence": avg_conf,
                "occurrences": count
            })

            db.add(AudioLabelSummary(
                run_id=run.id,
                label=label,
                avg_confidence=avg_conf,
                count=count
            ))

        summary.sort(key=lambda x: x["avg_confidence"], reverse=True)

        await db.commit()

        return {
            "engine": "yamnet-diagnostic",
            "run_id": str(run.id),
            "file": audio_file.filename,
            "metadata": metadata,
            "duration": run.duration,
            "sample_rate": run.sample_rate,
            "windows": run.windows,
            "window_detections": window_outputs,
            "summary": summary[:20]
        }

    except Exception as e:
        await db.rollback()
        logger.exception("Diagnostic failed")
        raise HTTPException(status_code=500, detail=str(e))

FFMPEG_PATH = r"C:\Users\caio\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-6.1-full_build\bin"  # <-- adjust if needed


if FFMPEG_PATH not in os.environ["PATH"]:
    os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ["PATH"]

whisper_model = whisper.load_model("small")


@app.post("/api/transcribe")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    language: Optional[str] = Form("en"),
    db: AsyncSession = Depends(get_db)
):
    """
    Transcribe audio using Whisper + Diagnose audio using YAMNet
    """
    try:
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="Empty filename")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Received audio file: {audio_file.filename}")

        # ─────────────────────────────────────
        # 1️⃣ Save uploaded audio to temp path
        # ─────────────────────────────────────
        audio_bytes = await audio_file.read()
        suffix = Path(audio_file.filename).suffix or ".wav"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            audio_path = Path(tmp.name)

        logger.info(f"Saved temp audio file: {audio_path}")

        # ─────────────────────────────────────
        # 2️⃣ Whisper Transcription
        # ─────────────────────────────────────
        lang_map = {
            "en-us": "en", "en": "en", "english": "en",
            "hi": "hi", "hindi": "hi",
            "ur": "ur", "urdu": "ur"
        }
        whisper_lang = lang_map.get(language.lower(), "en")

        whisper_result = whisper_model.transcribe(
            str(audio_path),
            language=whisper_lang,
            fp16=False
        )

        transcription_text = whisper_result["text"]
        detected_language = whisper_result.get("language", "unknown")
        segments = whisper_result.get("segments", [])

        # ─────────────────────────────────────
        # 3️⃣ YAMNet Preprocessing
        # ─────────────────────────────────────
        data = preprocessor.preprocess(str(audio_path))
        metadata = extract_mp3_metadata(audio_path)

        run = AudioDiagnosticRun(
            filename=audio_file.filename,
            duration=round(data["duration"], 2),
            sample_rate=data["sample_rate"],
            windows=len(data["windows"]),
            metadata=metadata
        )
        db.add(run)
        await db.flush()  # get run.id

        all_detections = []
        window_outputs = []

        # ─────────────────────────────────────
        # 4️⃣ YAMNet Window Inference
        # ─────────────────────────────────────
        for idx, (window, start, end) in enumerate(data["windows"]):
            preds = yamnet.predict_window(window, top_k=20)

            filtered = [
                {"label": label, "confidence": float(conf)}
                for label, conf in preds
                if conf >= 0.05
            ]

            window_row = AudioWindowDetection(
                run_id=run.id,
                window_index=idx + 1,
                start_time=round(start, 2),
                end_time=round(end, 2)
            )
            db.add(window_row)
            await db.flush()

            for p in filtered:
                db.add(AudioWindowLabel(
                    window_id=window_row.id,
                    label=p["label"],
                    confidence=p["confidence"]
                ))
                all_detections.append((p["label"], p["confidence"]))

            window_outputs.append({
                "window_index": idx + 1,
                "start_time": round(start, 2),
                "end_time": round(end, 2),
                "predictions": filtered
            })

        # ─────────────────────────────────────
        # 5️⃣ Aggregate Label Summary
        # ─────────────────────────────────────
        label_map = {}
        for label, conf in all_detections:
            label_map.setdefault(label, []).append(conf)

        summary = []
        for label, scores in label_map.items():
            avg_conf = round(sum(scores) / len(scores), 3)
            count = len(scores)

            summary.append({
                "label": label,
                "confidence": avg_conf,
                "occurrences": count
            })

            # db.add(AudioLabelSummary(
            #     run_id=run.id,
            #     label=label,
            #     avg_confidence=avg_conf,
            #     count=count
            # ))

        summary.sort(key=lambda x: x["confidence"], reverse=True)

        await db.commit()

        # ─────────────────────────────────────
        # 6️⃣ Cleanup temp audio
        # ─────────────────────────────────────
        os.remove(audio_path)

        # ─────────────────────────────────────
        # 7️⃣ Unified Response
        # ─────────────────────────────────────
        return {
            "success": True,
            "engine": ["whisper", "yamnet"],
            "run_id": str(run.id),
            "file": audio_file.filename,
            "metadata": metadata,
            "audio": {
                "duration": run.duration,
                "sample_rate": run.sample_rate,
                "windows": run.windows
            },
            "transcription": {
                "text": transcription_text,
                "segments": segments,
                "detected_language": detected_language,
                "requested_language": language
            },
            "yamnet": {
                "window_detections": window_outputs,
                "summary": summary[:20]
            }
        }

    except Exception as e:
        await db.rollback()
        logger.exception("Unified transcription + diagnosis failed")
        raise HTTPException(status_code=500, detail=str(e))
