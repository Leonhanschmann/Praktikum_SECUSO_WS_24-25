"""
Processors package for the gaze tracker application.
Provides data processing and state management components.
"""

# Import classes from gaze_processor.py
from .gaze_processor import (
    GazeProcessor,
    GazePoint
)

# Import TargetProcessor from target_processor.py
from .target_processor import (
    TargetProcessor
)

# Import classes from gaze_analyzer.py
from .gaze_analyzer import (
    GazeAnalyzer,
    Fixation,
    Saccade
)

# Export the processor classes
__all__ = [
    'GazeProcessor',
    'GazePoint',
    'TargetProcessor',
    'GazeAnalyzer',
    'Fixation',
    'Saccade'
]
