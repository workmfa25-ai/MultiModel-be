# 🎯 YAMNet Quick Start Guide

## Why YAMNet Instead of BEATs?

✅ **Only 5MB** (vs 500MB for BEATs)  
✅ **Auto-downloads** - no manual download needed  
✅ **521 sound classes** - detects same sounds  
✅ **Faster** inference  
✅ **Works immediately** - no download hassles  

---

## Setup (One-time)

```bash
cd sound_detection
.\sounds\Scripts\activate
pip install tensorflow tensorflow-hub
```

**Note:** TensorFlow is ~330MB but downloads automatically

---

## Usage

### Detect Sounds in Audio File

```bash
.\sounds\Scripts\activate
python detect_yamnet.py --input ..\audio_files\audio1.mp3
```

### Process All Files

```bash
python detect_yamnet.py --input ..\audio_files\audio1.mp3
python detect_yamnet.py --input ..\audio_files\audio_bengali.mp3
python detect_yamnet.py --input ..\audio_files\audio_chinese.mp3
```

---

## What You'll Get

Same output as BEATs:
- **JSON Report** - Full detection data
- **CSV Report** - For Excel/analysis  
- **Text Report** - Human-readable summary
- **Audio Segments** - WAV files of detected events

**Output location:** `output/reports/` and `output/segments/`

---

## Comparison: YAMNet vs BEATs

| Feature | YAMNet | BEATs |
|---------|--------|-------|
| Model Size | 5 MB | 500 MB |
| Download | Auto | Manual |
| Sound Classes | 521 | 632 |
| Accuracy | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Speed | Very Fast | Fast |
| Setup Difficulty | Easy | Hard |

**Recommendation:** Use YAMNet - it's easier and works great!

---

## Example Output

```
============================================================
Sound Detection with YAMNet
============================================================
Audio: audio1.mp3
============================================================

[1/4] Preprocessing audio...
Loaded audio: 120.50s @ 16000Hz

[2/4] Running YAMNet detection...
Processing 14 windows...

[3/4] Post-processing results...
Final: 8 events

[4/4] Saving results...

============================================================
Detection Complete!
============================================================
Total events detected: 8

By category:
  gunshots: 2
  vehicle_sounds: 4
  alarms: 2

Top detections:
  - gunshot at 15.3s (confidence: 89%)
  - car_door at 45.2s (confidence: 85%)
  - siren at 78.5s (confidence: 91%)

Results saved to: output/reports/
============================================================
```

---

## Next Steps

Once TensorFlow finishes installing:

```bash
# Test YAMNet
python detect_yamnet.py --input ..\audio_files\audio1.mp3

# Check results
type output\reports\audio1_report.txt

# Listen to detected segments
start output\segments\audio1_001_gunshot_15.3s.wav
```

---

## Troubleshooting

### "No module named 'tensorflow'"
```bash
.\sounds\Scripts\activate
pip install tensorflow tensorflow-hub
```

### "Model download failed"
- Check internet connection
- Model auto-downloads on first use (5MB)
- Takes ~10-30 seconds

### "No detections found"
- Lower thresholds in `config/sound_classes.py`
- YAMNet uses same config as BEATs

---

**YAMNet is ready to use - no manual downloads needed!** 🎉
