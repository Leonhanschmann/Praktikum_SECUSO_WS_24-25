# =========================================================================================
# IMAGE_TASK_VIEW.PY
#
# Provides the ImageTaskView class, which handles the image task phase by displaying a
# sequence of images while tracking and visualizing gaze data. It manages the timing and
# sequence logic, displays gaze overlays, and ensures smooth transitions between images.
# =========================================================================================

import time
from collections import deque
from datetime import datetime
from typing import Deque, List, Optional, Tuple
import pygame
from .base_view import BaseView


class ImageTaskView(BaseView):
    """
    Handles the image task phase, displaying a sequence of images while tracking gaze,
    with all timing and sequence logic encapsulated within the class.
    
    This view manages the presentation of each image for a specified duration, overlays gaze
    visualization (including trails and glow effects), and handles optional fading between images.
    It also displays minimal debug information such as gaze coordinates, remaining images, and
    the current system time.
    """

    def __init__(self, screen: pygame.Surface, width: int, height: int):
        """
        Initialize the ImageTaskView instance with necessary properties and surfaces.
        
        Args:
            screen (pygame.Surface): The main display surface where all drawings occur.
            width (int): Width of the display in pixels.
            height (int): Height of the display in pixels.
        """
        super().__init__(screen, width, height)

        # Gaze overlay settings
        self.GAZE_RADIUS = 20               # Radius of the gaze glow effect
        self.TRAIL_LENGTH = 25              # Number of previous gaze points to display as a trail
        self.GAZE_COLOR = (64, 196, 255)    # Color of the gaze visualization (light blue)
        self.show_gaze_overlay: bool = True # Flag to toggle gaze overlay visibility
        # Surface for drawing gaze visuals with per-pixel alpha for transparency
        self.gaze_surface = pygame.Surface((width, height), pygame.SRCALPHA)

        # Image sequence timing
        self.IMAGE_DISPLAY_TIME = 2.0        # Duration to display each image (in seconds)
        self.image_paths: List[str] = []     # List of image file paths to display
        self.current_idx = 0                  # Index of the currently displayed image
        self.sequence_complete = False        # Flag indicating if the image sequence is complete
        self.display_start_time: Optional[float] = None  # Timestamp when the current image started displaying

        # Current image properties
        self.current_image: Optional[pygame.Surface] = None  # Pygame surface of the current image
        self.image_rect: Optional[pygame.Rect] = None        # Rectangle defining the current image's position and size

        # Optional fade logic (currently unused but can be extended for fade transitions)
        self.fade_alpha = 255               # Alpha value for fade effect (255 = fully opaque)
        self.FADE_SPEED = 5                 # Speed at which the fade effect occurs (unused)

    def set_images(self, image_paths: List[str]) -> None:
        """
        Initialize the image sequence with a list of file paths.
        
        This method loads the first image in the sequence and prepares it for display.
        If no images are provided, the sequence is marked as complete immediately.
        
        Args:
            image_paths (List[str]): List of file paths to the images to be displayed.
        """
        self.image_paths = image_paths
        self.current_idx = 0
        self.sequence_complete = False

        if not self.image_paths:
            # No images provided; mark the sequence as complete
            self.sequence_complete = True
            return

        # Load and prepare the first image in the sequence
        self._load_current_image()
        self.display_start_time = time.time()  # Record the start time for the image display

    def _load_current_image(self) -> None:
        """
        Load the current image from file and scale it to fit the screen.
        
        This method loads the image at the current index, scales it to fit within the
        display dimensions while maintaining aspect ratio, and centers it on the screen.
        If an image fails to load, it is skipped, and the current image properties are cleared.
        """
        if self.current_idx >= len(self.image_paths):
            # Current index exceeds available images; clear current image properties
            self.current_image = None
            self.image_rect = None
            return

        image_path = self.image_paths[self.current_idx]
        try:
            # Load the image from the specified file path
            original = pygame.image.load(image_path)
            img_width = original.get_width()
            img_height = original.get_height()

            # Calculate scaling factor to fit the image within the display while maintaining aspect ratio
            width_ratio = self.width / img_width
            height_ratio = self.height / img_height
            scale_factor = min(width_ratio, height_ratio)

            new_width = int(img_width * scale_factor)
            new_height = int(img_height * scale_factor)

            # Smoothly scale the image to the new dimensions
            self.current_image = pygame.transform.smoothscale(
                original, (new_width, new_height)
            )

            # Center the image on the screen
            x = (self.width - new_width) // 2
            y = (self.height - new_height) // 2
            self.image_rect = pygame.Rect(x, y, new_width, new_height)

        except pygame.error as e:
            # Log an error message if the image fails to load and clear current image properties
            print(f"Error loading image {image_path}: {e}")
            self.current_image = None
            self.image_rect = None

    def check_image_complete(self) -> bool:
        """
        Determine if the current image's display time has elapsed.
        
        This method compares the elapsed time since the image started displaying with the
        predefined IMAGE_DISPLAY_TIME.
        
        Returns:
            bool: True if the display time has elapsed; False otherwise.
        """
        if self.sequence_complete or self.display_start_time is None:
            # Sequence is complete or display has not started; no completion to check
            return False

        # Calculate the elapsed time since the image started displaying
        time_elapsed = time.time() - self.display_start_time
        return time_elapsed >= self.IMAGE_DISPLAY_TIME

    def next_image(self) -> None:
        """
        Advance to the next image in the sequence or mark the sequence as complete if at the end.
        
        This method increments the current image index, loads the next image, and resets
        relevant properties. If the end of the image list is reached, the sequence is marked
        as complete.
        """
        self.current_idx += 1
        if self.current_idx >= len(self.image_paths):
            # Reached the end of the image sequence; mark as complete
            self.sequence_complete = True
            self.current_image = None
            self.image_rect = None
            return

        # Load and prepare the next image in the sequence
        self._load_current_image()
        self.display_start_time = time.time()  # Reset the display start time for the new image
        self.fade_alpha = 255                 # Reset fade alpha (currently unused)

    def is_sequence_complete(self) -> bool:
        """
        Check if the image sequence has been fully displayed.
        
        Returns:
            bool: True if all images have been displayed; False otherwise.
        """
        return self.sequence_complete

    def remaining_images(self) -> int:
        """
        Calculate the number of images remaining in the sequence, including the current one.
        
        Returns:
            int: The number of remaining images.
        """
        return max(0, len(self.image_paths) - self.current_idx)

    def reset(self) -> None:
        """
        Reset the ImageTaskView for a fresh run of the image sequence.
        
        This method resets the image index, marks the sequence as incomplete, and clears
        the display start time. If images are available, it loads the first image and
        records the new display start time.
        """
        self.current_idx = 0
        self.sequence_complete = False
        self.display_start_time = None
        self.fade_alpha = 255

        if self.image_paths:
            # Load the first image and record the display start time
            self._load_current_image()
            self.display_start_time = time.time()

    def draw(
        self,
        current_gaze: Optional[Tuple[int, int]],
        gaze_history: Deque[Tuple[int, int]],
        remaining_images: int,
    ) -> None:
        """
        Render the current image with optional gaze visualization.
        
        This method clears the screen, draws the current image, overlays the heatmap if available,
        and renders gaze visualization (trail and glow) if enabled. It also displays minimal
        debug information such as gaze coordinates, remaining images, and the current system time.
        
        Args:
            current_gaze (Optional[Tuple[int, int]]): The latest smoothed gaze position.
            gaze_history (Deque[Tuple[int, int]]): A deque containing the history of recent gaze positions.
            remaining_images (int): The number of images left to display in the sequence.
        """
        # Clear the screen with the background color
        self.screen.fill(self.BACKGROUND_COLOR)

        # Draw the current image if one is loaded
        if self.current_image and self.image_rect:
            self.screen.blit(self.current_image, self.image_rect)

        # Draw gaze visualization if enabled and a valid gaze position is available
        if self.show_gaze_overlay and current_gaze:
            self.draw_gaze(current_gaze, gaze_history)

        # Draw minimal debug information (gaze coordinates, remaining images, current time)
        self.draw_debug_info(current_gaze, remaining_images)

        # Update the entire display
        pygame.display.flip()

    def draw_gaze(
        self,
        current_gaze: Tuple[int, int],
        gaze_history: Deque[Tuple[int, int]],
    ) -> None:
        """
        Draw the gaze trail and glow effect on a transparent surface.
        
        This method visualizes the user's gaze by drawing a trail of previous gaze points
        and a glowing effect at the current gaze position. It utilizes anti-aliased circles
        and lines to create smooth visual effects.
        
        Args:
            current_gaze (Tuple[int, int]): The latest smoothed gaze position.
            gaze_history (Deque[Tuple[int, int]]): A deque containing the history of recent gaze positions.
        """
        # Clear the gaze surface to remove previous drawings
        self.clear_surface(self.gaze_surface)

        # Draw a trail of older gaze points if there are multiple points in history
        if len(gaze_history) > 1:
            points = list(gaze_history)
            for i in range(len(points) - 1):
                # Calculate progress along the trail for dynamic effects
                progress = i / (len(points) - 1)
                alpha = int(180 * progress)  # Fade out the trail
                size = int(self.GAZE_RADIUS * 0.5 * (1 - progress))  # Decrease size for older points

                # Draw smaller anti-aliased circles for the trail
                self.draw_aa_circle(
                    self.gaze_surface,
                    points[i],
                    size,
                    (*self.GAZE_COLOR, alpha),
                )
                # Draw lines connecting the trail points for continuity
                pygame.draw.line(
                    self.gaze_surface,
                    (*self.GAZE_COLOR, alpha),
                    points[i],
                    points[i + 1],
                    2,  # Line thickness
                )

        # Draw a glow effect at the current gaze position
        for radius in range(self.GAZE_RADIUS + 12, self.GAZE_RADIUS - 12, -1):
            alpha = int(130 * (radius / self.GAZE_RADIUS))
            self.draw_aa_circle(
                self.gaze_surface,
                current_gaze,
                radius,
                (*self.GAZE_COLOR, alpha),
            )

        # Blit the gaze surface onto the main screen to overlay the gaze visualization
        self.screen.blit(self.gaze_surface, (0, 0))

    def draw_debug_info(
        self,
        current_gaze: Optional[Tuple[int, int]],
        remaining_images: int,
    ) -> None:
        """
        Display minimal debug information on the screen.
        
        This includes the current gaze coordinates, the number of remaining images in the
        sequence, and the current system time. The information is displayed at predefined
        positions on the screen for easy visibility.
        
        Args:
            current_gaze (Optional[Tuple[int, int]]): The latest smoothed gaze position.
            remaining_images (int): The number of images left to display in the sequence.
        """
        if current_gaze:
            # Format the gaze coordinates for display
            gaze_text = f"X: {current_gaze[0]:4d}  Y: {current_gaze[1]:4d}"
        else:
            # Display 'NaN' if no valid gaze position is available
            gaze_text = "X: NaN  Y: NaN"

        # Render and position the gaze coordinates at (20, 20) pixels
        self.draw_text(gaze_text, (20, 20))
        # Render and position the remaining images count at (20, 60) pixels
        self.draw_text(f"Remaining images: {remaining_images}", (20, 60))

        # Get the current system time in a readable format
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Render and position the current time near the top-right corner
        self.draw_text(current_time, (self.width - 300, 20))

    def toggle_gaze_overlay(self) -> None:
        """
        Toggle the visibility of the gaze overlay.
        
        This method switches the `show_gaze_overlay` flag between True and False,
        allowing users to show or hide the gaze visualization on the images.
        """
        self.show_gaze_overlay = not self.show_gaze_overlay
