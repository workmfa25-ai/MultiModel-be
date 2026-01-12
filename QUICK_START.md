# 🎯 Quick Start - 3 Simple Steps

## Step 1: Open CMD and Navigate to Folder
```cmd
cd c:\Users\Asus\Downloads\new_audio_backend+audio_files\sound_detection
```

## Step 2: Run Detection
```cmd
run_detection.bat ..\audio_files\audio1.mp3
```

## Step 3: View Results
```cmd
type output\reports\audio1_report.txt
```

---

## That's It! 🎉

### To detect different files:
```cmd
run_detection.bat ..\audio_files\11.mp3
run_detection.bat ..\audio_files\audio_bengali.mp3
```

### To see ALL sounds (not just target sounds):
```cmd
.\sounds\Scripts\activate
python show_all_sounds.py --input ..\audio_files\audio1.mp3
```

---

## 📁 Where to Find Results

- **Reports**: `output\reports\`
  - `audio1_report.txt` - Summary
  - `audio1_detections.json` - Full data
  - `audio1_detections.csv` - Excel format

- **Audio Clips**: `output\segments\`
  - Individual WAV files of detected sounds

---

## ⚙️ Adjust Sensitivity

Edit `config\sound_classes.py` and change the numbers:

- **Lower numbers** (e.g., 0.20) = More detections
- **Higher numbers** (e.g., 0.50) = Fewer, more accurate detections

Current settings: 0.30 - 0.40 (balanced)

---

**Need help? Check README.md for full documentation!**
