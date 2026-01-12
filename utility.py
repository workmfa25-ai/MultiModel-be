from mutagen import File as MutagenFile
from mutagen.id3 import ID3
from pathlib import Path
import datetime


def extract_mp3_metadata(file_path: Path):
    metadata = {
        "file": file_path.name,
        "file_size_bytes": file_path.stat().st_size,
        "modified_time": datetime.datetime.fromtimestamp(
            file_path.stat().st_mtime
        ).isoformat()
    }

    audio = MutagenFile(file_path, easy=True)
    raw = MutagenFile(file_path)

    if audio:
        metadata["tags"] = {
            "title": audio.get("title", [None])[0],
            "artist": audio.get("artist", [None])[0],
            "album": audio.get("album", [None])[0],
            "genre": audio.get("genre", [None])[0],
            "date": audio.get("date", [None])[0],
        }

    if raw and raw.info:
        metadata["audio_properties"] = {
            "duration_seconds": round(raw.info.length, 2),
            "bitrate_kbps": getattr(raw.info, "bitrate", None),
            "sample_rate": getattr(raw.info, "sample_rate", None),
            "channels": getattr(raw.info, "channels", None),
            "codec": raw.mime[0] if raw.mime else None
        }

    return metadata
