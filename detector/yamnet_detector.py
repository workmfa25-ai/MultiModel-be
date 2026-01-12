"""
YAMNet Sound Detector
Lightweight alternative to BEATs - auto-downloads, only 5MB
"""

import numpy as np
from pathlib import Path
import logging
from typing import List, Dict
import tensorflow as tf
import tensorflow_hub as hub

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YAMNetDetector:
    """Sound event detector using Google YAMNet model"""
    
    def __init__(self):
        """Initialize YAMNet detector"""
        logger.info("Loading YAMNet model (auto-downloading if needed)...")
        
        # Load YAMNet model from TensorFlow Hub (auto-downloads)
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        
        # Load class names
        self.class_names = self._load_class_names()
        
        logger.info(f"YAMNet model loaded with {len(self.class_names)} classes")
    
    def _load_class_names(self):
        """Load YAMNet class names"""
        import csv
        
        class_map_path = self.model.class_map_path().numpy().decode('utf-8')
        class_names = []
        
        with open(class_map_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 3:
                    # Row format: index, mid, display_name
                    class_names.append(row[2])  # Display name
        
        return class_names
    
    def predict_window(self, audio_window, sample_rate=16000, top_k=10):
        """
        Predict sound events for a single audio window
        
        Args:
            audio_window (np.array): Audio window
            sample_rate (int): Sample rate (YAMNet expects 16kHz)
            top_k (int): Return top K predictions
            
        Returns:
            list: List of (label, confidence) tuples
        """
        try:
            # YAMNet expects float32 audio
            if audio_window.dtype != np.float32:
                audio_window = audio_window.astype(np.float32)
            
            # Run inference
            scores, embeddings, spectrogram = self.model(audio_window)
            
            # Get mean scores across all frames
            mean_scores = np.mean(scores.numpy(), axis=0)
            
            # Get top K predictions
            top_indices = np.argsort(mean_scores)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                label = self.class_names[idx]
                confidence = float(mean_scores[idx])
                results.append((label, confidence))
            
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return []
    
    def predict_audio(self, audio, sample_rate, windows_info, top_k=10):
        """
        Predict sound events for entire audio with windowing
        
        Args:
            audio (np.array): Full audio array
            sample_rate (int): Sample rate
            windows_info (list): List of (window_audio, start_time, end_time)
            top_k (int): Top K predictions per window
            
        Returns:
            list: List of detections with timestamps
        """
        all_detections = []
        
        logger.info(f"Processing {len(windows_info)} windows...")
        
        for idx, (window, start_time, end_time) in enumerate(windows_info):
            # Get predictions for this window
            predictions = self.predict_window(window, sample_rate, top_k)
            
            # Add timestamp information
            for label, confidence in predictions:
                all_detections.append({
                    'window_id': idx,
                    'label': label,
                    'confidence': confidence,
                    'start_time': start_time,
                    'end_time': end_time,
                    'window_center': (start_time + end_time) / 2
                })
        
        logger.info(f"Generated {len(all_detections)} raw detections")
        return all_detections
