"""
Sound Event Detection - Target Sound Classes Configuration
"""

# Target sound categories and their associated labels
TARGET_SOUNDS = {
    "gunshots": [
        "gunshot",
        "gunfire",
        "gunshot,_gunfire",
        "machine_gun",
        "explosion",
        "firecracker",
        "cap_gun",
        "fireworks"
    ],
    
    "vehicle_sounds": [
        "car",
        "vehicle",
        "motor_vehicle_(road)",
        "car_passing_by",
        "vehicle_horn",
        "car_alarm",
        "engine_starting",
        "engine",
        "medium_engine_(mid_frequency)",
        "light_engine_(high_frequency)",
        "heavy_engine_(low_frequency)",
        "engine_knocking",
        "engine_acceleration",
        "revving,_vroom",
        "door_slam",
        "slam"
    ],
    
    "alerts_and_alarms": [
        "alarm",
        "smoke_detector",
        "fire_alarm",
        "siren",
        "civil_defense_siren",
        "ambulance_(siren)",
        "police_car_(siren)",
        "fire_engine_(siren)",
        "emergency_vehicle",
        "buzzer",
        "alarm_clock"
    ],
    
    "impacts_and_crashes": [
        "crash",
        "breaking",
        "glass",
        "shatter",
        "bang",
        "boom",
        "thump",
        "thunk",
        "smash",
        "crash_cymbal",
        "wood_crack"
    ],
    
    "human_distress": [
        "scream",
        "screaming",
        "shout",
        "yell",
        "crying",
        "sobbing",
        "whimper",
        "gasp"
    ]
}

# Confidence threshold for each category
CONFIDENCE_THRESHOLDS = {
    "gunshots": 0.20,           # Very low to catch all potential gunshots
    "vehicle_sounds": 0.25,     # Low for better vehicle detection
    "alerts_and_alarms": 0.25,  # Low for alarm detection
    "impacts_and_crashes": 0.25,
    "human_distress": 0.25,
    "default": 0.20             # Very low default threshold
}

# Audio processing parameters
AUDIO_CONFIG = {
    "sample_rate": 16000,        # Target sample rate for model
    "window_size": 10.0,         # Seconds per analysis window
    "overlap": 2.0,              # Overlap between windows (seconds)
    "min_event_duration": 0.01,  # Very low minimum duration (10ms)
    "max_event_duration": 10.0,  # Increased max duration
}

# Model configuration
MODEL_CONFIG = {
    "model_name": "YAMNet",
    "auto_download": True,         # YAMNet auto-downloads from TensorFlow Hub
    "device": "cpu",               # YAMNet uses CPU (TensorFlow)
    "batch_size": 8,
}

# Output configuration
OUTPUT_CONFIG = {
    "save_segments": True,       # Save audio segments of detected events
    "segment_padding": 0.5,      # Padding around event (seconds)
    "export_json": True,         # Export results as JSON
    "export_csv": True,          # Export results as CSV
    "create_report": True,       # Create summary report
}

# Get all target sound labels as a flat list
def get_all_target_labels():
    """Returns a flat list of all target sound labels"""
    labels = []
    for category, sounds in TARGET_SOUNDS.items():
        labels.extend(sounds)
    return list(set(labels))  # Remove duplicates

# Get category for a given label
def get_category(label):
    """Returns the category for a given sound label"""
    label_lower = label.lower()
    for category, sounds in TARGET_SOUNDS.items():
        if any(sound.lower() in label_lower or label_lower in sound.lower() 
               for sound in sounds):
            return category
    return "other"

# Get confidence threshold for a label
def get_confidence_threshold(label):
    """Returns the confidence threshold for a given label"""
    category = get_category(label)
    return CONFIDENCE_THRESHOLDS.get(category, CONFIDENCE_THRESHOLDS["default"])
