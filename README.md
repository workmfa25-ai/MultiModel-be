# 🎵 Sound Event Detection System - YAMNet

Detects specific sound events (gunshots, vehicle sounds, alarms, etc.) from audio files using **Google's YAMNet** model.

---

## 🎯 What It Detects

The system detects **5 main categories** with **50+ sound types**:

1. **🔫 Gunshots & Explosions**
   - gunshot, gunfire, machine gun, explosion, firecracker

2. **🚗 Vehicle Sounds**
   - car door slam, engine sounds, horn, car alarm, vehicle passing

3. **🚨 Alerts & Alarms**
   - fire alarm, siren, emergency vehicle, smoke detector, buzzer

4. **💥 Impacts & Crashes**
   - glass breaking, crash, bang, smash, thump

5. **😱 Human Distress**
   - scream, shout, crying, sobbing, gasp

---

## 🚀 Quick Start

### Step 1: Activate Virtual Environment

```cmd
cd c:\Users\Asus\Downloads\new_audio_backend+audio_files\sound_detection
.\sounds\Scripts\activate
```

### Step 2: Install Dependencies (One-time)

```cmd
python -m pip install tensorflow tensorflow-hub
```

### Step 3: Run Detection

```cmd
python detect_yamnet.py --input ..\audio_files\audio1.mp3
```

**That's it!** YAMNet will auto-download (5MB) on first use and start detecting sounds.

---

## 💻 Usage

### Detect Sounds in Single File

```cmd
python detect_yamnet.py --input ..\audio_files\audio1.mp3
```

### See ALL Sounds Detected (Not Just Target Sounds)

```cmd
python show_all_sounds.py --input ..\audio_files\audio1.mp3
```

### Show More Sounds with Lower Confidence

```cmd
python show_all_sounds.py --input ..\audio_files\audio1.mp3 --top-k 30 --min-confidence 0.05
```

---

## 📊 Output Format

After detection, you'll find results in the `output/` folder:

### Reports (`output/reports/`)
- **JSON** - `audio1_detections.json` - Full detection data
- **CSV** - `audio1_detections.csv` - Excel-friendly format
- **TXT** - `audio1_report.txt` - Human-readable summary

### Audio Segments (`output/segments/`)
- Extracted audio clips of detected events
- Example: `audio1_001_gunshot_15.3s.wav`

---

## � Example Output

```
============================================================
Sound Detection with YAMNet
============================================================
Audio: audio1.mp3
============================================================

[1/4] Preprocessing audio...
Loaded audio: 71.45s @ 16000Hz

[2/4] Running YAMNet detection...
Processing 9 windows...

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

## ⚙️ Configuration

Edit `config/sound_classes.py` to customize:

### Confidence Thresholds

```python
CONFIDENCE_THRESHOLDS = {
    "gunshots": 0.35,           # 35% confidence
    "vehicle_sounds": 0.30,     # 30% confidence
    "alerts_and_alarms": 0.40,  # 40% confidence
    "impacts_and_crashes": 0.35,
    "human_distress": 0.40,
    "default": 0.30
}
```

**Lower values** = More detections (but more false positives)  
**Higher values** = Fewer detections (but more accurate)

### Target Sounds

Add or remove sounds from the `TARGET_SOUNDS` dictionary.

---

## 🔧 Troubleshooting

### "No module named 'tensorflow'"
```cmd
.\sounds\Scripts\activate
python -m pip install tensorflow tensorflow-hub
```

### "Fatal error in launcher"
Use `python -m pip` instead of `pip`:
```cmd
python -m pip install tensorflow tensorflow-hub
```

### "No detections found"
1. Lower confidence thresholds in `config/sound_classes.py`
2. Use `show_all_sounds.py` to see what YAMNet actually detects
3. Check if audio contains target sounds

---

## � Available Scripts

| Script | Purpose |
|--------|---------|
| **detect_yamnet.py** | Main detection script |
| **show_all_sounds.py** | Show ALL detected sounds (unfiltered) |

---

## 🎓 How It Works

1. **Preprocessing** - Audio is loaded and split into 10-second windows
2. **Detection** - YAMNet analyzes each window (521 sound classes)
3. **Filtering** - Only target sounds above confidence threshold are kept
4. **Merging** - Overlapping detections are merged
5. **Export** - Results saved as JSON, CSV, and text reports
6. **Segments** - Audio clips of detected events are extracted

---

## ✨ Why YAMNet?

✅ **Only 5MB** (vs 500MB for other models)  
✅ **Auto-downloads** - no manual setup  
✅ **521 sound classes** - comprehensive coverage  
✅ **Fast** inference on CPU  
✅ **Easy to use** - works immediately  

---

## 📚 Technical Details

- **Model**: Google YAMNet (AudioSet-trained)
- **Framework**: TensorFlow 2.x
- **Input**: 16kHz mono audio
- **Window Size**: 10 seconds with 2-second overlap
- **Classes**: 521 AudioSet sound event classes

---

## 🎯 Quick Command Reference

```cmd
# Setup (one-time)
cd c:\Users\Asus\Downloads\new_audio_backend+audio_files\sound_detection
.\sounds\Scripts\activate
python -m pip install tensorflow tensorflow-hub

# Detect sounds
python detect_yamnet.py --input ..\audio_files\audio1.mp3

# See all sounds
python show_all_sounds.py --input ..\audio_files\audio1.mp3

# View results
type output\reports\audio1_report.txt
```

---

**Ready to detect sounds! 🎧**