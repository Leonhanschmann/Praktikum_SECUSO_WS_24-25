#views/heatmap_view.py

# =========================================================================================
# HEATMAP_VIEW.PY
#
# Provides a view for rendering a heatmap of gaze data, optionally with
# target markers. The heatmap is generated in a background thread to avoid blocking
# the main application loop. Includes a loading screen to display generation progress,
# and a legend for interpreting the heatmap’s color scale.
# =========================================================================================

import pygame
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from .base_view import BaseView
import threading


@dataclass
class GazePoint:
    """
    Represents a single gaze point measurement.

    Attributes:
        timestamp (float): The Unix timestamp when this gaze point was recorded.
        position (Tuple[int,int]): The (x, y) screen coordinates of the gaze.
        velocity (float): Estimated gaze velocity in pixels/second.
    """
    timestamp: float
    position: Tuple[int, int]
    velocity: float


class HeatmapView(BaseView):
    """
    Renders a heatmap representation of gaze data. Uses a background thread to build
    a density map, allowing the main application to remain responsive.
    """

    def __init__(self, screen: pygame.Surface, width: int, height: int):
        """
        Initialize the HeatmapView with display dimensions and default settings.

        Args:
            screen (pygame.Surface): The main Pygame surface to draw on.
            width (int): Screen width in pixels.
            height (int): Screen height in pixels.
        """
        super().__init__(screen, width, height)
        
        # Color overrides
        self.BACKGROUND_COLOR = (20, 20, 30)
        self.GRID_COLOR = (40, 40, 60)
        self.TARGET_COLOR = (255, 165, 0)  # Orange
        self.TEXT_COLOR = (255, 255, 255)
        self.FONT = pygame.font.SysFont("Arial", 20)
        
        # Heatmap granularity and Gaussian settings
        self.GRID_SIZE = 2       # Each grid cell represents 2x2 pixels
        self.KERNEL_SIZE = 101   # Not directly used here, but can inform kernel-based methods
        self.SIGMA = 50.0        # Standard deviation for Gaussian contributions
        
        # Surfaces for layering the heatmap
        self.heatmap_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.cached_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.has_cached_render = False  # Tracks whether we have a valid cached surface
        
        # Threading and progress tracking
        self.thread_lock = threading.Lock()
        self.generation_thread = None
        self.should_stop = False
        self.is_generating = False
        self.current_progress = 0
        
        # Internal data for density map generation
        self.density_map = None
        self.ready_to_draw = False
        self.positions_to_process = None

    # ------------------------------------------------------------------------------------
    # Internal Helper Methods
    # ------------------------------------------------------------------------------------
    def _intensity_to_color(self, intensity: float) -> Tuple[int, int, int, int]:
        """
        Map a normalized intensity value [0..1] to a color gradient (blue -> red) with
        intermediate cyan, green, and yellow.

        Args:
            intensity (float): Value between 0 and 1, representing heatmap intensity.

        Returns:
            (r, g, b, a): RGBA color tuple.
        """
        intensity = np.clip(intensity, 0, 1)
        
        # Define color stops for the gradient
        colors = [
            (0, (0, 0, 255)),      # Blue
            (0.25, (0, 255, 255)), # Cyan
            (0.5, (0, 255, 0)),    # Green
            (0.75, (255, 255, 0)), # Yellow
            (1, (255, 0, 0))       # Red
        ]
        
        # Interpolate between color stops
        for i in range(len(colors) - 1):
            if intensity <= colors[i + 1][0]:
                t = (intensity - colors[i][0]) / (colors[i + 1][0] - colors[i][0])
                c1 = colors[i][1]
                c2 = colors[i + 1][1]
                
                r = int(c1[0] + t * (c2[0] - c1[0]))
                g = int(c1[1] + t * (c2[1] - c1[1]))
                b = int(c1[2] + t * (c2[2] - c1[2]))
                alpha = int(255 * min(1.0, intensity * 1.5))
                
                return (r, g, b, alpha)
                
        # Fallback to the last color stop
        return colors[-1][1] + (255,)

    def _generate_density_map(self):
        """
        Background thread function to create a 2D density map from the gaze point positions,
        then apply a Gaussian filter for smoothing. Finally, updates the cached surface.
        """
        try:
            with self.thread_lock:
                # If there's no data or we've been told to stop, exit early
                if self.should_stop or not self.positions_to_process:
                    return
                    
                # Create a grid where each cell corresponds to GRID_SIZE x GRID_SIZE pixels
                grid = np.zeros((self.height // self.GRID_SIZE,
                                 self.width // self.GRID_SIZE))
                
                # For indexing
                y_indices, x_indices = np.mgrid[0:grid.shape[0], 0:grid.shape[1]]
                total_points = len(self.positions_to_process)
                
                # Aggregate Gaussian contributions from each point
                for idx, (x, y) in enumerate(self.positions_to_process):
                    if self.should_stop:
                        return

                    # Update generation progress
                    self.current_progress = (idx + 1) / total_points
                    
                    # Convert pixel coords to grid coords
                    grid_x = x / self.GRID_SIZE
                    grid_y = y / self.GRID_SIZE
                    
                    # Gaussian formula
                    gaussian = np.exp(-((x_indices - grid_x) ** 2 + (y_indices - grid_y) ** 2)
                                      / (2.0 * (self.SIGMA / self.GRID_SIZE) ** 2))
                    grid += gaussian
                
                if self.should_stop:
                    return
                
                # Normalize values to [0..1]
                if grid.max() > 0:
                    grid /= grid.max()
                    
                # Optionally apply further smoothing
                from scipy.ndimage import gaussian_filter
                self.density_map = gaussian_filter(grid, sigma=2.0)
                
                # Render the final map to the cached surface
                self._render_to_cache()
                self.ready_to_draw = True
                
        finally:
            self.is_generating = False

    def _render_to_cache(self):
        """
        Convert the internal density map into colored pixels and store in self.cached_surface.
        """
        self.cached_surface.fill((0, 0, 0, 0))
        
        if self.density_map is not None:
            for y in range(self.density_map.shape[0]):
                for x in range(self.density_map.shape[1]):
                    val = self.density_map[y, x]
                    if val > 0.01:  # Skip very low intensity to reduce overdraw
                        color = self._intensity_to_color(val)
                        rect = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
                        rect.fill(color)
                        self.cached_surface.blit(rect, (x * self.GRID_SIZE, y * self.GRID_SIZE))
        self.has_cached_render = True

    def _draw_loading_screen(self):
        """
        Display a loading message and progress bar while the heatmap is still generating.
        """
        self.screen.fill(self.BACKGROUND_COLOR)
        
        text = self.FONT.render("Generating Heatmap...", True, self.TEXT_COLOR)
        text_rect = text.get_rect(center=(self.width // 2, self.height // 2 - 50))
        self.screen.blit(text, text_rect)
        
        # Progress bar
        bar_width = 400
        bar_height = 20
        border_rect = pygame.Rect(self.width // 2 - bar_width // 2,
                                  self.height // 2,
                                  bar_width,
                                  bar_height)
        progress_rect = pygame.Rect(border_rect.left,
                                    border_rect.top,
                                    border_rect.width * self.current_progress,
                                    bar_height)
        
        pygame.draw.rect(self.screen, (100, 100, 100), border_rect, 2)
        pygame.draw.rect(self.screen, (100, 100, 100), progress_rect)
        
        # Percentage text
        percent_text = self.FONT.render(f"{int(self.current_progress * 100)}%", True, self.TEXT_COLOR)
        percent_rect = percent_text.get_rect(center=(self.width // 2, self.height // 2 + 50))
        self.screen.blit(percent_text, percent_rect)
        
        pygame.display.flip()

    # ------------------------------------------------------------------------------------
    # Public Methods
    # ------------------------------------------------------------------------------------
    def draw_targets(self,
                     target_positions: List[Tuple[int, int]],
                     completion_times: Optional[List[float]] = None) -> None:
        """
        Optionally draws target markers and completion times over the heatmap.

        Args:
            target_positions (List[Tuple[int,int]]): List of target (x, y) coordinates.
            completion_times (Optional[List[float]]): Completion times for each target.
        """
        for i, pos in enumerate(target_positions):
            pygame.draw.circle(self.screen, self.TARGET_COLOR, pos, 15, 2)
            pygame.draw.circle(self.screen, self.TARGET_COLOR, pos, 10, 2)
            pygame.draw.circle(self.screen, self.TARGET_COLOR, pos, 3)
            
            if completion_times and i < len(completion_times):
                label_surf = self.FONT.render(f"{completion_times[i]:.1f}s", True, self.TEXT_COLOR)
                self.screen.blit(label_surf, (pos[0] + 20, pos[1] - 10))

    def draw_grid(self) -> None:
        """
        Draw a background grid to provide spatial reference.
        """
        for x in range(0, self.width, 50):
            pygame.draw.line(self.screen, self.GRID_COLOR, (x, 0), (x, self.height))
        for y in range(0, self.height, 50):
            pygame.draw.line(self.screen, self.GRID_COLOR, (0, y), (self.width, y))

    def start_generation(self, gaze_points: List[GazePoint]) -> None:
        """
        Initiate heatmap generation in a background thread.

        Args:
            gaze_points (List[GazePoint]): All the gaze points to be used for heatmap creation.
        """
        positions = [point.position for point in gaze_points]
        if not positions:
            return
            
        with self.thread_lock:
            if not self.is_generating and self.density_map is None:
                self.positions_to_process = positions
                self.is_generating = True
                self.ready_to_draw = False
                self.current_progress = 0
                self.should_stop = False
                self.generation_thread = threading.Thread(target=self._generate_density_map)
                self.generation_thread.start()

    def draw(self,
             gaze_points: List[GazePoint],
             target_positions: List[Tuple[int, int]],
             completion_times: Optional[List[float]] = None) -> None:
        """
        Main draw method for updating the heatmap view.

        Args:
            gaze_points (List[GazePoint]): Not directly used once generation is done,
                                           but kept for consistency with other views.
            target_positions (List[Tuple[int,int]]): Coordinates of targets to overlay.
            completion_times (Optional[List[float]]): Optional times to show next to targets.
        """
        # If the heatmap is still generating, draw the loading screen
        if self.is_generating:
            self._draw_loading_screen()
            return

        # If the heatmap is ready, draw it
        if self.ready_to_draw and self.has_cached_render:
            self.screen.fill(self.BACKGROUND_COLOR)
            self.draw_grid()
            self.screen.blit(self.cached_surface, (0, 0))
            self.draw_targets(target_positions, completion_times)
            self.draw_legend()  # <--- CALL THE LEGEND DRAWING HERE
            pygame.display.flip()
        else:
            # Draw a blank background if we have no data
            self.screen.fill(self.BACKGROUND_COLOR)
            self.draw_grid()
            pygame.display.flip()

    def reset(self) -> None:
        """
        Reset the view state, stopping any ongoing generation and clearing progress.
        """
        with self.thread_lock:
            # Signal any running thread to stop
            self.should_stop = True

            # If a thread is running, wait briefly for it to join
            if self.generation_thread and self.generation_thread.is_alive():
                self.generation_thread.join(timeout=0.1)

            # Clear flags and references
            self.is_generating = False
            self.current_progress = 0
            self.generation_thread = None

    def clear(self) -> None:
        """
        Completely clear all data, including the generated heatmap.
        This should be used when starting a new session or discarding old data.
        """
        self.reset()
        self.density_map = None
        self.ready_to_draw = False
        self.has_cached_render = False
        self.positions_to_process = None

    # ------------------------------------------------------------------------------------
    # Legend Drawing
    # ------------------------------------------------------------------------------------
    def draw_legend(self) -> None:
        """
        Draw a  legend box indicating how color intensities map from 'Low' to 'High'.
        """
        legend_w = 160
        legend_h = 70
        margin = 20

        # Position the legend in the bottom-right corner
        x = self.width - legend_w - margin
        y = self.height - legend_h - margin
        rect = pygame.Rect(x, y, legend_w, legend_h)

        # Draw the panel background
        pygame.draw.rect(self.screen, (30, 30, 50), rect)
        pygame.draw.rect(self.screen, self.GRID_COLOR, rect, 1)

        # Title
        label = self.FONT.render("Gaze Intensity", True, self.TEXT_COLOR)
        lx = rect.centerx - (label.get_width() // 2)
        ly = rect.top + 5
        self.screen.blit(label, (lx, ly))

        # Define the gradient area
        grad_left = rect.left + 10
        grad_right = rect.right - 10
        grad_top = ly + label.get_height() + 5
        grad_bottom = rect.bottom - 20
        grad_height = grad_bottom - grad_top
        grad_width = grad_right - grad_left

        if grad_width <= 0:
            return

        # Draw the color gradient from 0.0 (blue) to 1.0 (red)
        for i in range(grad_width):
            ratio = i / float(grad_width)
            color = self._intensity_to_color(ratio)
            # Draw a thin vertical line for the gradient
            pygame.draw.line(self.screen, color, (grad_left + i, grad_top), (grad_left + i, grad_bottom))

        # "Low" label on the left
        low_surf = self.FONT.render("Low", True, self.TEXT_COLOR)
        self.screen.blit(low_surf, (grad_left, grad_bottom + 2))

        # "High" label on the right
        high_surf = self.FONT.render("High", True, self.TEXT_COLOR)
        self.screen.blit(high_surf, (grad_right - high_surf.get_width(), grad_bottom + 2))
