"""
Sound Detection Module - Initialization
"""

from .sound_classes import (
    TARGET_SOUNDS,
    CONFIDENCE_THRESHOLDS,
    AUDIO_CONFIG,
    MODEL_CONFIG,
    OUTPUT_CONFIG,
    get_all_target_labels,
    get_category,
    get_confidence_threshold
)

__all__ = [
    'TARGET_SOUNDS',
    'CONFIDENCE_THRESHOLDS',
    'AUDIO_CONFIG',
    'MODEL_CONFIG',
    'OUTPUT_CONFIG',
    'get_all_target_labels',
    'get_category',
    'get_confidence_threshold'
]
