# processors/target_processor.py

# =========================================================================================
# TARGET_PROCESSOR.PY
#
# Manages the creation and handling of sequential targets for the dot task.
# Each target must be fixated on for a specific duration before it fades and
# the next one appears. Both absolute and relative completion times are recorded/stored.
# =========================================================================================


import time
import numpy as np
from typing import List, Tuple, Optional


class TargetProcessor:
    """
    Orchestrates target positions for a fixation task and manages each target’s timing
    and completion state. Each target requires a minimum gaze duration before being
    considered 'complete'. Completion times are tracked in both absolute and relative forms.
    """

    def __init__(self, screen_width: int, screen_height: int):
        """
        Initialize the TargetProcessor with specified screen dimensions.

        Args:
            screen_width (int): The width of the display in pixels.
            screen_height (int): The height of the display in pixels.
        """
        self.width = screen_width
        self.height = screen_height
        
        # Target tracking
        self.positions: List[Tuple[int, int]] = []  # (x, y) coordinates for each target
        self.current_idx = 0                       # Index of the current target to be fixated
        self.visible_time: Optional[float] = None  # Timestamp when gaze first landed on the current target
        
        # Display attributes
        self.alpha = 255                 # Current alpha for the fading animation
        self.size_multiplier = 1.0       # Controls the scaling of the dot in animations
        self.target_size = 1.0           # Baseline target size, used for animated transitions
        
        # Timing settings
        self.FIXATION_TIME = 0.8         # Required continuous fixation time in seconds
        self.GAZE_PERIMETER = 60         # Pixel distance threshold to consider gaze "on-target"
        
        # Completion time tracking
        self.completion_times_absolute: List[float] = []  # List of absolute timestamps of target completions
        self.completion_times: List[float] = []           # Times relative to self.start_time
        self.start_time: Optional[float] = None           # Start time for computing relative times

    def generate_positions(self, n_positions: int) -> None:
        """
        Generate random target positions within the screen, ensuring they
        are kept some distance from the screen edges. Reset states and completion info.

        Args:
            n_positions (int): How many distinct target positions to create.
        """
        margin = 100
        self.positions = []
        for _ in range(n_positions):
            x = np.random.randint(margin, self.width - margin)
            y = np.random.randint(margin, self.height - margin)
            self.positions.append((x, y))

        # Reset task progress
        self.current_idx = 0
        self.visible_time = None
        self.alpha = 255
        self.size_multiplier = 1.0
        self.target_size = 1.0
        self.start_time = None
        
        # Clear old completion times
        self.completion_times_absolute.clear()
        self.completion_times.clear()

    def check_gaze(self, gaze_pos: Optional[Tuple[int, int]]) -> bool:
        """
        Check if the user’s gaze is on the current target. If so, track fixation time
        and begin fade-out when it exceeds FIXATION_TIME. Completes the target once fully faded.

        Args:
            gaze_pos (Optional[Tuple[int,int]]): The current (x, y) position of the user’s gaze.
                                                 If None, gaze is unavailable.

        Returns:
            bool: True if all targets are completed; False otherwise.
        """
        if not gaze_pos or self.current_idx >= len(self.positions):
            # Either no gaze data or we've run out of targets
            return False
            
        dot_pos = self.positions[self.current_idx]
        dx = gaze_pos[0] - dot_pos[0]
        dy = gaze_pos[1] - dot_pos[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        # If gaze is within the threshold => fixating on target
        if distance <= self.GAZE_PERIMETER:
            self.target_size = 1.3  # Grow the target slightly as a feedback indicator

            if self.visible_time is None:
                # Gaze has just arrived on the target
                self.visible_time = time.time()
                if self.start_time is None:
                    self.start_time = self.visible_time
            else:
                # Check if the user has maintained fixation long enough
                if time.time() - self.visible_time >= self.FIXATION_TIME:
                    # Start the fade-out effect
                    self.alpha = max(0, self.alpha - 5)
                    if self.alpha == 0:
                        # Target fully faded => mark completion
                        completion_time = time.time()
                        self.completion_times_absolute.append(completion_time)

                        # Record relative completion time, fallback to 0 if no start_time
                        if self.start_time is not None:
                            relative_time = completion_time - self.start_time
                            self.completion_times.append(relative_time)
                        else:
                            self.completion_times.append(0.0)
        
                        self.current_idx += 1
                        self.visible_time = None
                        self.alpha = 255
                        self.size_multiplier = 1.0
                        self.target_size = 1.0

                        # If no more targets remain, we're done
                        if self.current_idx >= len(self.positions):
                            return True
        else:
            # If gaze is off-target, reset timing and revert size
            self.visible_time = None
            self.target_size = 1.0
            
        return False

    def update_animation(self) -> None:
        """
        Smoothly animate target size transitions. Moves the size_multiplier closer to 
        target_size via a simple proportional step each frame.
        """
        if abs(self.target_size - self.size_multiplier) > 0.01:
            self.size_multiplier += (self.target_size - self.size_multiplier) * 0.1

    def reset(self) -> None:
        """
        Reset the TargetProcessor’s internal state, including completion records,
        alpha, size scaling, and index for the current target.
        """
        self.current_idx = 0
        self.visible_time = None
        self.alpha = 255
        self.size_multiplier = 1.0
        self.target_size = 1.0
        self.start_time = None
        
        # Clear any existing completion data
        self.completion_times_absolute.clear()
        self.completion_times.clear()