#views/verification_view.py


# =========================================================================================
# VERIFICATION_VIEW.PY
#
# Provides a view for conducting the dot task. Displays a target dot
# for the user to fixate on, along with a real-time gaze point and trail for visualization.
# Also includes debug information like gaze coordinates and timestamp.
# =========================================================================================



from datetime import datetime
from typing import Tuple, Optional, Deque
import pygame
from .base_view import BaseView


class VerificationView(BaseView):
    """
    Handles the verification phase of gaze tracking, drawing a target dot and
    user gaze in real time. Used for conducting the dot task.
    """

    def __init__(self, screen: pygame.Surface, width: int, height: int):
        """
        Initialize the verification view with display properties and visuals.

        Args:
            screen (pygame.Surface): The main pygame surface to draw on.
            width (int): Screen width in pixels.
            height (int): Screen height in pixels.
        """
        super().__init__(screen, width, height)
        
        # Verification-specific constants
        self.DOT_RADIUS = 15
        self.GAZE_RADIUS = 20
        self.TRAIL_LENGTH = 25
        self.FIXATION_COLOR = (0, 255, 0, 128)  # Semi-transparent green
        self.TARGET_COLOR = (255, 165, 0)       # Orange for target dots
        
        # Internal state
        self.fixation_points = []  # Unused in this snippet but reserved for possible expansions

    def draw(self,
             current_gaze: Optional[Tuple[int, int]],
             dot_position: Optional[Tuple[int, int]],
             dot_alpha: int,
             dot_size_multiplier: float,
             gaze_history: Deque[Tuple[int, int]],
             remaining_points: int) -> None:
        """
        Render the entire verification view, including the target dot, user gaze,
        a gaze trail, and any debug or textual info.

        Args:
            current_gaze (Optional[Tuple[int,int]]): Current gaze coordinates, or None if invalid.
            dot_position (Optional[Tuple[int,int]]): The (x, y) position of the target dot, if any.
            dot_alpha (int): Alpha (0-255) for target dot fading effects.
            dot_size_multiplier (float): Dynamic scaling factor for the target dot’s radius.
            gaze_history (Deque[Tuple[int,int]]): Recent gaze positions for trail visualization.
            remaining_points (int): Number of targets left in the verification or calibration sequence.
        """
        # Clear the background
        self.screen.fill(self.BACKGROUND_COLOR)
        
        # Draw the target dot if it’s visible/active
        if dot_position:
            self.draw_dot(dot_position, dot_alpha, dot_size_multiplier)
        
        # Draw the gaze indicator and trail if valid gaze data is present
        if current_gaze:
            self.draw_gaze(current_gaze, gaze_history)
        
        # Display helpful debug information (coordinates, etc.)
        self.draw_debug_info(current_gaze, remaining_points)
        
        # Present the updated frame
        pygame.display.flip()

    def draw_dot(self,
                 position: Tuple[int, int],
                 alpha: int,
                 size_multiplier: float) -> None:
        """
        Render a target dot with a glowing effect. The dot can fade or grow
        depending on external logic (alpha, size_multiplier).

        Args:
            position (Tuple[int,int]): (x, y) location of the dot.
            alpha (int): Transparency (0-255) for the dot’s fade-in/out effect.
            size_multiplier (float): Scaling factor applied to the baseline dot radius.
        """
        self.clear_surface(self.dot_surface)
        current_radius = self.DOT_RADIUS * size_multiplier
        
        # Outer glow: draw multiple circles with varying alpha to create a glow effect
        for radius in range(int(current_radius) + 8, int(current_radius) - 4, -1):
            alpha_mod = int(alpha * (radius / (current_radius + 8)) * 0.8)
            self.draw_aa_circle(
                self.dot_surface,
                position,
                radius,
                (*self.DOT_COLOR, alpha_mod)
            )
        
        # Core solid dot
        self.draw_aa_circle(
            self.dot_surface,
            position,
            max(1, current_radius - 2),  # Ensure radius doesn't go below 1
            (*self.DOT_COLOR, alpha)
        )
        
        self.screen.blit(self.dot_surface, (0, 0))

    def draw_gaze(self,
                  current_gaze: Tuple[int, int],
                  gaze_history: Deque[Tuple[int, int]]) -> None:
        """
        Draw the current gaze position along with a trailing history to visualize
        recent movement.

        Args:
            current_gaze (Tuple[int,int]): Latest gaze coordinates on-screen.
            gaze_history (Deque[Tuple[int,int]]): Historical gaze positions.
        """
        self.clear_surface(self.gaze_surface)
        
        # Draw the gaze trail if enough points exist
        if len(gaze_history) > 1:
            points = list(gaze_history)
            for i in range(len(points) - 1):
                # Determine alpha and size based on position along the trail
                progress = i / (len(points) - 1)
                alpha = int(180 * progress)    # Increases alpha as we go further back in history
                size = int(self.GAZE_RADIUS * 0.5 * (1 - progress))
                
                # Draw a circle representing this point in the trail
                self.draw_aa_circle(
                    self.gaze_surface,
                    points[i],
                    size,
                    (*self.GAZE_COLOR, alpha)
                )
                
                # Draw a line to the next point in the trail
                pygame.draw.line(
                    self.gaze_surface,
                    (*self.GAZE_COLOR, alpha),
                    points[i],
                    points[i + 1],
                    2
                )
        
        # Draw the current gaze position with a glowing effect
        for radius in range(self.GAZE_RADIUS + 12, self.GAZE_RADIUS - 12, -1):
            alpha = int(130 * (radius / self.GAZE_RADIUS))
            self.draw_aa_circle(
                self.gaze_surface,
                current_gaze,
                radius,
                (*self.GAZE_COLOR, alpha)
            )
        
        # Blit the gaze overlay
        self.screen.blit(self.gaze_surface, (0, 0))

    def draw_debug_info(self,
                        current_gaze: Optional[Tuple[int, int]],
                        remaining_points: int) -> None:
        """
        Display textual debug info, such as the user’s gaze coordinates
        and how many calibration points remain.

        Args:
            current_gaze (Optional[Tuple[int,int]]): Current gaze position, or None if unavailable.
            remaining_points (int): Number of remaining calibration points.
        """
        # Show gaze coordinates (or placeholder if gaze is invalid)
        if current_gaze:
            gaze_text = f"X: {current_gaze[0]:4d}  Y: {current_gaze[1]:4d}"
        else:
            gaze_text = "X: NaN  Y: NaN"
        self.draw_text(gaze_text, (20, 20))
        
        # Show the remaining points count
        self.draw_text(f"Remaining points: {remaining_points}", (20, 60))
        
        # Display the current timestamp
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.draw_text(current_date, (self.width - 300, 20))





