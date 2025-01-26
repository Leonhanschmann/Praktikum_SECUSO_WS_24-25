# processors/gaze_processor.py

# =========================================================================================
# GAZE_PROCESSOR.PY
#
# Provides real-time gaze data capture, including optional position smoothing
# for user-friendly visualization. Raw points are stored for later analysis.
# =========================================================================================

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Deque
import numpy as np
import time


@dataclass
class GazePoint:
    """
    Represents a single gaze point measurement.

    Attributes:
        timestamp (float): The Unix timestamp of when this point was recorded.
        position (Tuple[int,int]): (x, y) pixel coordinates on screen.
        velocity (float): Gaze movement velocity in pixels/second.
    """
    timestamp: float
    position: Tuple[int, int]
    velocity: float


class GazeProcessor:
    """
    Manages incoming gaze data for real-time visualization and later analysis.

    The GazeProcessor computes a smoothed gaze position for display (avoid noisy jumps)
    and accumulates raw gaze data for advanced analysis. When recording is active,
    new data is processed and stored; otherwise, data is ignored.
    """

    def __init__(self, screen_width: int, screen_height: int):
        """
        Initialize a new GazeProcessor instance.

        Args:
            screen_width (int): Width of the display in pixels.
            screen_height (int): Height of the display in pixels.
        """
        # Screen dimensions
        self.width = screen_width
        self.height = screen_height
        
        # Smoothing factor for visual display
        self.SMOOTHING_FACTOR = 0.15
        
        # Visualization buffers
        self.gaze_history: Deque[Tuple[int, int]] = deque(maxlen=25)  # Keeps recent positions for a "trail"
        self.current_gaze: Optional[Tuple[int, int]] = None           # Latest smoothed gaze position
        self.smoothed_gaze: Optional[Tuple[float, float]] = None      # Internal state for position smoothing
        
        # Timing and data collection
        self.last_gaze_time: Optional[float] = None
        self.raw_gaze_points: List[GazePoint] = []  # All raw points for offline analysis
        self.is_recording = True                    # Recording state flag

    def smooth_position(self,
                        current: Tuple[float, float],
                        target: Tuple[float, float]) -> Tuple[float, float]:
        """
        Smoothly interpolate from 'current' to 'target' using a basic exponential filter.

        Args:
            current (Tuple[float,float]): The previous smoothed gaze position.
            target (Tuple[float,float]): The new raw gaze position to blend toward.

        Returns:
            Tuple[float,float]: The updated (smoothed) gaze position.
        """
        x1, y1 = current
        x2, y2 = target
        smooth_x = x1 + (x2 - x1) * self.SMOOTHING_FACTOR
        smooth_y = y1 + (y2 - y1) * self.SMOOTHING_FACTOR
        return (smooth_x, smooth_y)

    def calculate_velocity(self,
                           pos1: Tuple[int, int],
                           pos2: Tuple[int, int],
                           dt: float) -> float:
        """
        Compute the velocity of gaze movement between two points over the time interval dt.

        Args:
            pos1 (Tuple[int,int]): The previous gaze coordinates.
            pos2 (Tuple[int,int]): The current gaze coordinates.
            dt (float): The elapsed time in seconds between these two points.

        Returns:
            float: The velocity in pixels/second. Returns 0 if dt <= 0.
        """
        if dt <= 0:
            return 0.0
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        distance = np.sqrt(dx*dx + dy*dy)
        return distance / dt

    def process_gaze_data(self, gaze_data: Dict) -> None:
        """
        Process and store a single sample of gaze data when recording is active.
        Averages left and right eye positions to form a single (x, y) point.

        Args:
            gaze_data (Dict):
                A dictionary typically containing:
                {
                  'left_gaze_point_on_display_area': (lx, ly),
                  'right_gaze_point_on_display_area': (rx, ry)
                }
                Each coordinate is in the [0, 1] range, relative to the display.
        """
        if not self.is_recording:
            return
        
        left_eye = gaze_data.get('left_gaze_point_on_display_area')
        right_eye = gaze_data.get('right_gaze_point_on_display_area')
        current_time = time.time()

        # If both eyes have valid coordinates in the [0,1] range, compute a raw gaze position
        if left_eye and right_eye:
            left_x, left_y = left_eye
            right_x, right_y = right_eye
            
            # Validate all coords are within normalized range
            if all(0.0 <= coord <= 1.0 for coord in [left_x, left_y, right_x, right_y]):
                # Convert normalized [0,1] to screen pixel coordinates
                x = (left_x + right_x) / 2 * self.width
                y = (left_y + right_y) / 2 * self.height
                raw_gaze = (int(x), int(y))
                
                # Compute velocity based on the last known gaze position and time
                raw_velocity = 0.0
                if self.last_gaze_time is not None:
                    dt = current_time - self.last_gaze_time
                    if self.raw_gaze_points:
                        prev_point = self.raw_gaze_points[-1].position
                        raw_velocity = self.calculate_velocity(prev_point, raw_gaze, dt)
                
                # Store the new raw gaze point
                self.raw_gaze_points.append(
                    GazePoint(timestamp=current_time, position=raw_gaze, velocity=raw_velocity)
                )
                
                # Update smoothed gaze for display
                if self.smoothed_gaze is None:
                    self.smoothed_gaze = raw_gaze
                else:
                    self.smoothed_gaze = self.smooth_position(self.smoothed_gaze, raw_gaze)
                
                # Convert smoothed coords to integers for drawing
                self.current_gaze = (int(self.smoothed_gaze[0]), int(self.smoothed_gaze[1]))
                self.gaze_history.append(self.current_gaze)
                self.last_gaze_time = current_time
        else:
            # If eye data is incomplete, no valid current gaze
            self.current_gaze = None

    def stop_recording(self) -> None:
        """
        Deactivate recording. Future gaze data will be discarded until recording is re-enabled.
        """
        self.is_recording = False

    def start_recording(self) -> None:
        """
        Reactivate recording if it was previously stopped.
        """
        self.is_recording = True

    def reset(self) -> None:
        """
        Reset all internal states, clearing both the history and stored raw data.

        Call this before starting a new session to remove old data and states.
        """
        self.gaze_history.clear()
        self.current_gaze = None
        self.smoothed_gaze = None
        self.last_gaze_time = None
        self.raw_gaze_points.clear()
        self.is_recording = True