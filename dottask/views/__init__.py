"""
Views package for the gaze tracker application.
Provides different visualization modes for eye tracking data.
"""

from .base_view import BaseView
from .verification_view import VerificationView
from .analysis_view import AnalysisView
from .heatmap_view import HeatmapView  # Import HeatmapView

# Export the view classes
__all__ = [
    'BaseView',
    'VerificationView',
    'AnalysisView',
    'HeatmapView'  # Add HeatmapView to the export list
]
