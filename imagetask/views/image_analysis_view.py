# =========================================================================================
# IMAGE_ANALYSIS_VIEW.PY
#
# Provides the ImageAnalysisView class, which displays and manages heatmaps overlaid on their
# corresponding images. Once all images are shown in the image task phase, this view is presented.
# It generates each heatmap in a background thread using a ThreadPoolExecutor and displays a
# loading screen until generation is complete. Users can navigate between heatmaps using the
# LEFT and RIGHT arrow keys.
# =========================================================================================

import pygame
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import time

from .base_view import BaseView


@dataclass
class GazePoint:
    """
    Represents a single gaze point measurement.
    
    Attributes:
        timestamp (float): The Unix timestamp when this gaze point was recorded.
        position (Tuple[int, int]): (x, y) screen coordinates of the gaze.
        velocity (float): Estimated gaze velocity in pixels per second.
    """
    timestamp: float            # The Unix timestamp when this gaze point was recorded
    position: Tuple[int, int]   # (x, y) screen coordinates of the gaze
    velocity: float             # Estimated gaze velocity in pixels/second


@dataclass
class ImageHeatmapData:
    """
    Contains the heatmap data and state for a single image.
    
    Attributes:
        image_path (str): File path to the image.
        image_surface (pygame.Surface): The scaled Pygame surface of the image.
        image_rect (pygame.Rect): The rectangle defining the image's position and size on the screen.
        gaze_points (List[GazePoint]): List of gaze points associated with the image.
        density_map (Optional[np.ndarray]): 2D array representing the density of gaze points.
        heatmap_surface (Optional[pygame.Surface]): Pygame surface representing the rendered heatmap.
    """
    image_path: str
    image_surface: pygame.Surface
    image_rect: pygame.Rect
    gaze_points: List[GazePoint]
    density_map: Optional[np.ndarray] = None
    heatmap_surface: Optional[pygame.Surface] = None


class ImageAnalysisView(BaseView):
    """
    Displays and manages heatmaps overlaid on their corresponding images.
    
    Once all images are shown in the image task phase, this view is presented. It generates
    each heatmap in a background thread using a ThreadPoolExecutor and displays a loading screen
    until all heatmaps are generated. Users can navigate between heatmaps using the LEFT and RIGHT
    arrow keys. The view also includes a color legend to interpret heatmap intensities.
    """

    def __init__(self, screen: pygame.Surface, width: int, height: int):
        """
        Initialize the ImageAnalysisView instance.
        
        Args:
            screen (pygame.Surface): The main display surface where all drawings occur.
            width (int): Width of the display in pixels.
            height (int): Height of the display in pixels.
        """
        super().__init__(screen, width, height)

        # Override default colors and font for this specific view
        self.BACKGROUND_COLOR = (20, 20, 30)  # Dark bluish background
        self.TEXT_COLOR = (255, 255, 255)     # White text color
        self.GRID_COLOR = (40, 40, 60)        # Darker grid lines
        self.FONT = pygame.font.SysFont("Arial", 20)  # Arial font with size 20

        # Heatmap generation settings
        self.GRID_SIZE = 2       # Each cell in the grid corresponds to 2x2 pixels
        self.SIGMA = 50.0        # Standard deviation for Gaussian smoothing

        # ThreadPoolExecutor for generating heatmaps in background threads
        self.thread_pool: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=4)
        self.thread_lock = threading.Lock()  # Lock to manage access to shared resources

        # Navigation and data tracking
        self.current_image_index = 0  # Index of the currently displayed image
        self.image_data: List[ImageHeatmapData] = []  # List of ImageHeatmapData instances
        self.is_generating = False    # Flag indicating if heatmap generation is in progress
        self.generation_progress = 0.0  # Progress of heatmap generation (0.0 to 1.0)
        self.total_generations = 0      # Total number of heatmaps to generate
        self.completed_generations = 0  # Number of heatmaps generated so far

        # Navigation hint display settings
        self.show_navigation_hint = True  # Flag to show/hide navigation hints
        self.HINT_DURATION = 3000         # Duration to display navigation hint (in milliseconds)
        self.hint_start_time: Optional[float] = None  # Timestamp when hint was first shown

    # -------------------------------------------------------------------------
    # LOAD & DATA PREPARATION
    # -------------------------------------------------------------------------
    def load_images_and_data(
        self,
        image_paths: List[str],
        gaze_data_per_image: List[List[Tuple[int, int]]]
    ) -> None:
        """
        Load and prepare images along with their corresponding gaze points.
        
        This method processes each image by loading it, scaling it to fit the display,
        and associating it with its corresponding gaze points. Gaze points are converted
        into GazePoint instances with stub values for timestamp and velocity, as only
        positions are provided at this stage.
        
        Args:
            image_paths (List[str]): List of file paths to the images.
            gaze_data_per_image (List[List[Tuple[int, int]]]): List containing lists of (x, y)
                gaze positions for each image.
        """
        # Clear any existing image data
        self.image_data.clear()

        # Iterate over each image path and its corresponding gaze points
        for path, raw_points in zip(image_paths, gaze_data_per_image):
            # Create GazePoint instances with timestamp=0 and velocity=0
            gaze_points = [
                GazePoint(timestamp=0, position=(x, y), velocity=0) 
                for (x, y) in raw_points
            ]

            # Load the image from the file system
            original = pygame.image.load(path)
            img_w, img_h = original.get_width(), original.get_height()
            # Calculate scaling factor to fit the image within the display dimensions
            scale = min(self.width / img_w, self.height / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            # Smoothly scale the image to the new dimensions
            scaled_image = pygame.transform.smoothscale(original, (new_w, new_h))
            
            # Center the image on the screen
            x = (self.width - new_w) // 2
            y = (self.height - new_h) // 2
            image_rect = pygame.Rect(x, y, new_w, new_h)

            # Append the prepared ImageHeatmapData to the image_data list
            self.image_data.append(ImageHeatmapData(
                image_path=path,
                image_surface=scaled_image,
                image_rect=image_rect,
                gaze_points=gaze_points
            ))

        # Reset navigation and hint display settings
        self.current_image_index = 0
        self.show_navigation_hint = True
        self.hint_start_time = pygame.time.get_ticks()  # Record the start time for the hint

    # -------------------------------------------------------------------------
    # HEATMAP GENERATION
    # -------------------------------------------------------------------------
    def start_heatmap_generation(self) -> None:
        """
        Initiate background heatmap generation for all loaded images.
        
        This method checks if heatmap generation is already in progress or if there are
        no images to process. If the ThreadPoolExecutor has been shut down, it is
        re-initialized. Heatmap generation tasks are submitted to the executor for each
        image.
        """
        # Skip if generation is already in progress or there are no images to process
        if self.is_generating or not self.image_data:
            return

        # Re-initialize the ThreadPoolExecutor if it has been shut down
        if self.thread_pool is None:
            self.thread_pool = ThreadPoolExecutor(max_workers=4)

        # Set generation flags and counters
        self.is_generating = True
        self.generation_progress = 0.0
        self.total_generations = len(self.image_data)
        self.completed_generations = 0

        # Submit a heatmap generation task for each image
        for i in range(len(self.image_data)):
            self.thread_pool.submit(self._generate_heatmap, i)

    def _generate_heatmap(self, image_index: int) -> None:
        """
        Generate a heatmap for a single image in a background thread.
        
        This method processes the gaze points for the specified image to create a density map
        using Gaussian distribution. The density map is then smoothed and rendered to a Pygame
        surface for visualization.
        
        Args:
            image_index (int): The index of the image in the image_data list for which the
                heatmap is to be generated.
        """
        try:
            # Retrieve the ImageHeatmapData for the specified image
            data = self.image_data[image_index]
            # Extract all gaze positions for the image
            positions = [p.position for p in data.gaze_points]
            if not positions:
                # If there are no gaze points, increment completion counter and update progress
                with self.thread_lock:
                    self.completed_generations += 1
                    self._update_progress()
                return

            # Initialize a 2D grid representing the display area divided by GRID_SIZE
            grid = np.zeros((self.height // self.GRID_SIZE,
                             self.width // self.GRID_SIZE))

            # Create meshgrid indices for vectorized operations
            y_indices, x_indices = np.mgrid[0:grid.shape[0], 0:grid.shape[1]]

            # Accumulate Gaussian contributions from each gaze point
            for (x, y) in positions:
                grid_x = x / self.GRID_SIZE
                grid_y = y / self.GRID_SIZE
                # Calculate Gaussian value based on distance from the gaze point
                gaussian = np.exp(
                    -((x_indices - grid_x)**2 + (y_indices - grid_y)**2)
                    / (2.0 * (self.SIGMA / self.GRID_SIZE)**2)
                )
                grid += gaussian

            # Normalize the grid to ensure values are between 0 and 1
            if grid.max() > 0:
                grid /= grid.max()

            # Apply an additional Gaussian filter for smoother density distribution
            from scipy.ndimage import gaussian_filter
            data.density_map = gaussian_filter(grid, sigma=2.0)

            # Render the density map to a Pygame surface
            self._render_heatmap_surface(image_index)

            # Update the generation progress
            with self.thread_lock:
                self.completed_generations += 1
                self._update_progress()

        except Exception as e:
            # Log any errors that occur during heatmap generation
            print(f"Error generating heatmap for image {image_index}: {e}")

    def _update_progress(self) -> None:
        """
        Update the overall generation progress and determine if all heatmaps are generated.
        
        This method calculates the current progress based on the number of completed
        heatmap generations and updates the is_generating flag if all heatmaps are done.
        """
        self.generation_progress = (
            self.completed_generations / max(1, self.total_generations)
        )
        if self.completed_generations == self.total_generations:
            self.is_generating = False  # All heatmaps have been generated

    def _render_heatmap_surface(self, image_index: int) -> None:
        """
        Render the final density map to a colored Pygame surface.
        
        This method converts the normalized density map into a visual heatmap by mapping
        intensity values to colors and drawing them onto a Pygame surface.
        
        Args:
            image_index (int): The index of the image in the image_data list for which the
                heatmap is to be rendered.
        """
        data = self.image_data[image_index]
        if data.density_map is None:
            return  # No density map to render

        # Create a transparent surface for the heatmap
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        rows, cols = data.density_map.shape
        for y in range(rows):
            for x in range(cols):
                val = data.density_map[y, x]
                if val > 0.01:  # Skip very low values to reduce overdraw and improve performance
                    color = self._intensity_to_color(val)
                    # Create a small surface for each grid cell to represent intensity
                    rect = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
                    rect.fill(color)
                    # Blit the colored rectangle onto the heatmap surface at the appropriate position
                    surface.blit(rect, (x * self.GRID_SIZE, y * self.GRID_SIZE))

        # Assign the rendered heatmap surface to the corresponding ImageHeatmapData
        data.heatmap_surface = surface

    # -------------------------------------------------------------------------
    # DRAWING
    # -------------------------------------------------------------------------
    def draw(self) -> None:
        """
        Main draw method for the ImageAnalysisView.
        
        Displays a loading screen if heatmap generation is in progress. Otherwise, it shows
        the current image along with its corresponding heatmap. Additionally, it overlays
        navigation information and a color legend for interpreting heatmap intensities.
        """
        if self.is_generating:
            # Display the loading screen while heatmaps are being generated
            self._draw_loading_screen()
            return

        # If there are no images loaded, simply fill the screen with the background color
        if not self.image_data:
            self.screen.fill(self.BACKGROUND_COLOR)
            pygame.display.flip()
            return

        # Clear the screen with the background color
        self.screen.fill(self.BACKGROUND_COLOR)

        # Retrieve the current ImageHeatmapData to display
        current_data = self.image_data[self.current_image_index]
        # Draw the image onto the screen at its designated rectangle
        self.screen.blit(current_data.image_surface, current_data.image_rect)

        # Overlay the heatmap if it has been generated
        if current_data.heatmap_surface:
            self.screen.blit(current_data.heatmap_surface, (0, 0))

        # Overlay navigation information and heatmap legend
        self._draw_navigation_info()
        self._draw_legend()

        # Display a temporary navigation hint if applicable
        if self.show_navigation_hint:
            self._draw_navigation_hint()

        # Update the entire display
        pygame.display.flip()

    def _draw_loading_screen(self) -> None:
        """
        Display a loading screen with progress information during heatmap generation.
        
        This screen includes a title, a progress bar indicating the percentage of heatmaps
        generated, and a numerical percentage display.
        """
        # Clear the screen with the background color
        self.screen.fill(self.BACKGROUND_COLOR)

        # Render and position the title text
        title_surf = self.FONT.render("Generating Heatmaps...", True, self.TEXT_COLOR)
        title_rect = title_surf.get_rect(center=(self.width // 2, self.height // 2 - 50))
        self.screen.blit(title_surf, title_rect)

        # Define dimensions for the progress bar
        bar_w, bar_h = 400, 20
        border_rect = pygame.Rect(self.width // 2 - bar_w // 2, self.height // 2, bar_w, bar_h)
        progress_rect = pygame.Rect(
            border_rect.left,
            border_rect.top,
            border_rect.width * self.generation_progress,
            bar_h
        )

        # Draw the border of the progress bar
        pygame.draw.rect(self.screen, (120, 120, 120), border_rect, 2)
        # Fill the progress bar based on the current progress
        pygame.draw.rect(self.screen, (120, 120, 120), progress_rect)

        # Render and position the percentage text below the progress bar
        percent_text = f"{int(self.generation_progress * 100)}%"
        percent_surf = self.FONT.render(percent_text, True, self.TEXT_COLOR)
        percent_rect = percent_surf.get_rect(center=(self.width // 2, self.height // 2 + 50))
        self.screen.blit(percent_surf, percent_rect)

        # Update the entire display to show the loading screen
        pygame.display.flip()

    def _draw_navigation_info(self) -> None:
        """
        Display the current image index out of the total number of images.
        
        This information is shown at the top-left corner of the screen to inform the user
        which heatmap is currently being viewed.
        """
        info_text = f"Image {self.current_image_index + 1} of {len(self.image_data)}"
        text_surf = self.FONT.render(info_text, True, self.TEXT_COLOR)
        self.screen.blit(text_surf, (20, 20))  # Position at (20, 20) pixels

    def _draw_navigation_hint(self) -> None:
        """
        Display a temporary hint about using the arrow keys for navigation at the bottom of the screen.
        
        The hint is shown for a predefined duration (HINT_DURATION) and then hidden.
        """
        # Check if the hint has been displayed long enough to be hidden
        if self.hint_start_time:
            elapsed = pygame.time.get_ticks() - self.hint_start_time
            if elapsed > self.HINT_DURATION:
                self.show_navigation_hint = False  # Hide the hint after the duration
                return

        # Define the hint text
        hint_text = "Use LEFT/RIGHT arrow keys to navigate between heatmaps"
        # Render the hint text
        hint_surf = self.FONT.render(hint_text, True, self.TEXT_COLOR)
        # Position the hint at the bottom center of the screen
        hint_rect = hint_surf.get_rect(center=(self.width // 2, self.height - 40))
        self.screen.blit(hint_surf, hint_rect)

    def _draw_legend(self) -> None:
        """
        Draw a color scale legend for the heatmap intensity in the bottom-right corner.
        
        The legend provides a visual guide to interpret the color gradient used in the heatmaps,
        indicating the relationship between color intensity and gaze density.
        """
        # Define dimensions and positioning for the legend
        legend_w = 160
        legend_h = 80  # Increased height to accommodate labels
        margin = 20
        x = self.width - legend_w - margin
        y = self.height - legend_h - margin
        rect = pygame.Rect(x, y, legend_w, legend_h)

        # Draw the background rectangle for the legend
        pygame.draw.rect(self.screen, (30, 30, 50), rect)
        # Draw the border of the legend rectangle
        pygame.draw.rect(self.screen, self.GRID_COLOR, rect, 1)

        # Render and center the legend title
        label = self.FONT.render("Gaze Intensity", True, self.TEXT_COLOR)
        lx = rect.centerx - (label.get_width() // 2)
        ly = rect.top + 5
        self.screen.blit(label, (lx, ly))

        # Define the gradient area within the legend
        grad_left = rect.left + 10
        grad_right = rect.right - 10
        grad_top = ly + label.get_height() + 5
        grad_bottom = rect.bottom - 30  # Adjusted to provide space for labels
        grad_width = grad_right - grad_left

        if grad_width <= 0:
            return  # Exit if the gradient width is invalid

        # Draw the gradient color scale by mapping intensity to color
        for i in range(grad_width):
            ratio = i / float(grad_width)
            color = self._intensity_to_color(ratio)
            pygame.draw.line(self.screen, color,
                             (grad_left + i, grad_top),
                             (grad_left + i, grad_bottom))

        # Render "Low" and "High" labels within the legend area
        low_surf = self.FONT.render("Low", True, self.TEXT_COLOR)
        high_surf = self.FONT.render("High", True, self.TEXT_COLOR)

        # Position labels inside the legend to prevent them from being cut off
        self.screen.blit(low_surf, (grad_left, grad_bottom + 5))
        self.screen.blit(high_surf, (grad_right - high_surf.get_width(), grad_bottom + 5))

    # -------------------------------------------------------------------------
    # HANDLE INPUT
    # -------------------------------------------------------------------------
    def handle_input(self, event: pygame.event.Event) -> None:
        """
        Handle keyboard input for navigating between heatmaps.
        
        Users can navigate to the previous or next heatmap using the LEFT and RIGHT arrow keys.
        
        Args:
            event (pygame.event.Event): The event to handle.
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                # Navigate to the previous heatmap, ensuring the index doesn't go below 0
                self.current_image_index = max(0, self.current_image_index - 1)
            elif event.key == pygame.K_RIGHT:
                # Navigate to the next heatmap, ensuring the index doesn't exceed the list size
                self.current_image_index = min(
                    len(self.image_data) - 1,
                    self.current_image_index + 1
                )

    # -------------------------------------------------------------------------
    # COLOR MAPPING
    # -------------------------------------------------------------------------
    def _intensity_to_color(self, intensity: float) -> Tuple[int, int, int, int]:
        """
        Convert a normalized intensity value to a color on a blue-to-red gradient.
        
        This method maps intensity values between 0.0 and 1.0 to a gradient ranging from
        blue (low intensity) to red (high intensity), passing through cyan, green, and yellow.
        
        Args:
            intensity (float): A normalized value between 0.0 and 1.0 representing intensity.
        
        Returns:
            Tuple[int, int, int, int]: An RGBA color corresponding to the intensity.
        """
        # Clamp intensity to the [0.0, 1.0] range
        intensity = max(0.0, min(1.0, intensity))
        # Define color stops for the gradient
        colors = [
            (0, (0, 0, 255)),       # Blue
            (0.25, (0, 255, 255)),  # Cyan
            (0.5, (0, 255, 0)),     # Green
            (0.75, (255, 255, 0)),  # Yellow
            (1, (255, 0, 0))        # Red
        ]
        # Iterate through the color stops to find the appropriate segment for interpolation
        for i in range(len(colors) - 1):
            if intensity <= colors[i + 1][0]:
                # Calculate the interpolation factor within the current segment
                t = (intensity - colors[i][0]) / (colors[i + 1][0] - colors[i][0])
                c1 = colors[i][1]
                c2 = colors[i + 1][1]
                # Linearly interpolate each RGB component
                r = int(c1[0] + t * (c2[0] - c1[0]))
                g = int(c1[1] + t * (c2[1] - c1[1]))
                b = int(c1[2] + t * (c2[2] - c1[2]))
                # Adjust alpha based on intensity, capping at full opacity
                alpha = int(255 * min(1.0, intensity * 1.5))
                return (r, g, b, alpha)
        # Fallback color if intensity exceeds all defined stops (should not occur due to clamping)
        return (255, 0, 0, 255)

    # -------------------------------------------------------------------------
    # CLEANUP
    # -------------------------------------------------------------------------
    def cleanup(self) -> None:
        """
        Terminate any ongoing heatmap generation and clear all image data.
        
        If the ThreadPoolExecutor is active, it is shut down gracefully, waiting for all
        submitted tasks to complete. The image_data list is then cleared to free resources.
        """
        if self.thread_pool:
            # Shut down the ThreadPoolExecutor, waiting for all heatmaps to finish generating
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None  # Set to None to allow re-initialization if needed

        # Clear all image data to free memory
        self.image_data.clear()

    def reset(self) -> None:
        """
        Clear any data and reset the state of the ImageAnalysisView.
        
        This method calls the cleanup() function to terminate heatmap generation and clear
        data. It also resets generation flags and counters, preparing the view for a fresh
        analysis session.
        """
        self.cleanup()
        self.is_generating = False
        self.generation_progress = 0.0
        self.total_generations = 0
        self.completed_generations = 0
