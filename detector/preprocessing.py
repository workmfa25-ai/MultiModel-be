"""
Audio Preprocessing Module
Handles audio loading, resampling, and windowing for sound detection
"""

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Preprocesses audio files for sound event detection"""
    
    def __init__(self, target_sr=16000, window_size=10.0, overlap=2.0):
        """
        Initialize audio preprocessor
        
        Args:
            target_sr (int): Target sample rate in Hz
            window_size (float): Window size in seconds
            overlap (float): Overlap between windows in seconds
        """
        self.target_sr = target_sr
        self.window_size = window_size
        self.overlap = overlap
        self.hop_size = window_size - overlap
        
    def load_audio(self, audio_path):
        """
        Load audio file and convert to target sample rate
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            tuple: (audio_array, sample_rate, duration)
        """
        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            logger.info(f"Loading audio: {audio_path.name}")
            
            # Load audio with librosa (automatically converts to mono)
            audio, sr = librosa.load(str(audio_path), sr=self.target_sr, mono=True)
            
            duration = len(audio) / sr
            logger.info(f"Loaded audio: {duration:.2f}s @ {sr}Hz")
            
            return audio, sr, duration
            
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise
    
    def normalize_audio(self, audio):
        """
        Normalize audio to [-1, 1] range
        
        Args:
            audio (np.array): Audio array
            
        Returns:
            np.array: Normalized audio
        """
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def create_windows(self, audio, sr):
        """
        Split audio into overlapping windows
        
        Args:
            audio (np.array): Audio array
            sr (int): Sample rate
            
        Returns:
            list: List of (window_audio, start_time, end_time) tuples
        """
        window_samples = int(self.window_size * sr)
        hop_samples = int(self.hop_size * sr)
        
        windows = []
        start_sample = 0
        
        while start_sample < len(audio):
            end_sample = min(start_sample + window_samples, len(audio))
            
            # Extract window
            window = audio[start_sample:end_sample]
            
            # Pad if necessary (last window might be shorter)
            if len(window) < window_samples:
                window = np.pad(window, (0, window_samples - len(window)), mode='constant')
            
            start_time = start_sample / sr
            end_time = end_sample / sr
            
            windows.append((window, start_time, end_time))
            
            # Move to next window
            start_sample += hop_samples
            
            # Break if we've reached the end
            if end_sample >= len(audio):
                break
        
        logger.info(f"Created {len(windows)} windows")
        return windows
    
    def extract_segment(self, audio, sr, start_time, end_time, padding=0.5):
        """
        Extract a segment from audio with padding
        
        Args:
            audio (np.array): Full audio array
            sr (int): Sample rate
            start_time (float): Start time in seconds
            end_time (float): End time in seconds
            padding (float): Padding in seconds
            
        Returns:
            np.array: Audio segment
        """
        # Add padding
        start_time = max(0, start_time - padding)
        end_time = min(len(audio) / sr, end_time + padding)
        
        # Convert to samples
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        segment = audio[start_sample:end_sample]
        return segment
    
    def save_segment(self, segment, sr, output_path):
        """
        Save audio segment to file
        
        Args:
            segment (np.array): Audio segment
            sr (int): Sample rate
            output_path (str): Output file path
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            sf.write(str(output_path), segment, sr)
            logger.info(f"Saved segment: {output_path.name}")
            
        except Exception as e:
            logger.error(f"Error saving segment: {e}")
            raise
    
    def preprocess(self, audio_path):
        """
        Complete preprocessing pipeline
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            dict: Preprocessed data with audio, windows, and metadata
        """
        # Load audio
        audio, sr, duration = self.load_audio(audio_path)
        
        # Normalize
        audio = self.normalize_audio(audio)
        
        # Create windows
        windows = self.create_windows(audio, sr)
        
        return {
            'audio': audio,
            'sample_rate': sr,
            'duration': duration,
            'windows': windows,
            'file_name': Path(audio_path).name
        }
