#views/analysis_view.py


# =========================================================================================
# ANALYSIS_VIEW.PY
#
# Provides a comprehensive overview of recorded gaze data, including visualizations for
# gaze path, fixations, saccades, velocity profile, metrics, and more. Users can filter,
# hover, and interact with different components to explore data in detail.
# =========================================================================================




import pygame
import pygame.gfxdraw
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Union
from .base_view import BaseView
from .components import (
    GazePathVisualization,
    SaccadesVisualization,
    FixationsVisualization,
    TargetsVisualization,
    VelocityProfileVisualization,
    MetricsSummaryVisualization,
    TimelinePanel,
    FilterPanel
)


@dataclass
class GazePoint:
    """
    Represents a single gaze point measurement.

    Attributes:
        timestamp (float): The timestamp at which this gaze point was recorded.
        position (Tuple[int,int]): The (x, y) coordinates in screen pixels.
        velocity (float): Gaze movement velocity (px/s) at this point.
    """
    timestamp: float
    position: Tuple[int, int]
    velocity: float

@dataclass
class Fixation:
    """
    Represents a period of stable gaze.

    Attributes:
        start_time (float): The relative or analysis-based start time of the fixation.
        end_time (float): The relative or analysis-based end time of the fixation.
        center_position (Tuple[int,int]): The centroid of all points in this fixation.
        duration (float): Duration of the fixation in seconds.
        gaze_points (List[GazePoint]): The raw gaze points associated with the fixation.
        start_time_absolut (float): The absolute start timestamp (from raw data).
        end_time_absolut (float): The absolute end timestamp (from raw data).
    """
    start_time: float
    end_time: float
    center_position: Tuple[int, int]
    duration: float
    gaze_points: List[GazePoint]
    start_time_absolut: float
    end_time_absolut: float

@dataclass
class SaccadeSegment:
    """
    Represents a velocity-based sub-segment of a saccade (optional).

    Attributes:
        start_position (Tuple[int,int]): Where this segment of the saccade begins.
        end_position (Tuple[int,int]): Where this segment ends.
        mean_velocity (float): Mean velocity over this segment in px/s.
        peak_velocity (float): Peak velocity over this segment in px/s.
        gaze_point_count (int): Number of gaze points in this segment.
    """
    start_position: Tuple[int, int]
    end_position: Tuple[int, int]
    mean_velocity: float
    peak_velocity: float
    gaze_point_count: int

@dataclass
class Saccade:
    """
    Represents a rapid eye movement between fixations.

    Attributes:
        start_time (float): Relative or analysis-based start time of the saccade.
        end_time (float): Relative or analysis-based end time of the saccade.
        start_time_absolut (float): Absolute start timestamp (from raw data).
        end_time_absolut (float): Absolute end timestamp (from raw data).
        start_position (Tuple[int,int]): (x, y) of where the saccade begins.
        end_position (Tuple[int,int]): (x, y) of where the saccade ends.
        duration (float): Saccade duration in seconds.
        peak_velocity (float): Maximum velocity (px/s) within the saccade.
        mean_velocity (float): Average velocity over the entire saccade (px/s).
        amplitude (float): Euclidean distance between start_position and end_position.
        distance_traveled (float): Sum of consecutive distances (path length).
        gaze_points (List[GazePoint]): The individual gaze points forming this saccade.
        segments (Optional[List[SaccadeSegment]]): Optional velocity-segmented slices of the saccade.
    """
    start_time: float
    end_time: float
    start_time_absolut: float
    end_time_absolut: float
    start_position: Tuple[int, int]
    end_position: Tuple[int, int]
    duration: float
    peak_velocity: float
    mean_velocity: float
    amplitude: float
    distance_traveled: float
    gaze_points: List['GazePoint']
    segments: Optional[List[SaccadeSegment]] = None


# --------------------------------------------------------------------------------------
# AnalysisView
# --------------------------------------------------------------------------------------
class AnalysisView(BaseView):
    """
    Handles the graphical display of gaze data analyses, including:
      - Gaze path visualization
      - Fixation highlights
      - Saccade rendering
      - Velocity profile panel
      - Metrics summary panel
      - Filter and timeline panels
    """

    def __init__(self, screen: pygame.Surface, width: int, height: int):
        """
        Initialize the AnalysisView with all necessary sub-components and UI elements.

        Args:
            screen (pygame.Surface): The main Pygame surface to draw on.
            width (int): Screen width in pixels.
            height (int): Screen height in pixels.
        """
        super().__init__(screen, width, height)

        # Colors
        self.CHART_COLOR = (64, 196, 255)
        self.FIXATION_COLOR = (0, 128, 0)   # Olive green
        self.TARGET_COLOR = (255, 165, 0)  # Orange
        self.BACKGROUND_COLOR = (20, 20, 30)
        self.GRID_COLOR = (40, 40, 60)
        self.TOOLTIP_COLOR = (60, 60, 80)
        self.PANEL_COLOR = (30, 30, 40)
        self.BUTTON_COLOR = (40, 40, 60)
        self.BUTTON_BORDER_COLOR = (80, 80, 100)
        self.TEXT_COLOR = (255, 255, 255)
        self.FONT = pygame.font.SysFont("Arial", 20)

        # Visualization states
        self.velocity_profile_expanded = False
        self.metrics_expanded = False
        self.hover_fixation: Optional[Fixation] = None
        self.selected_fixation: Optional[Fixation] = None
        self.selected_saccade: Optional[Saccade] = None
        self.selected_target_index: Optional[int] = None
        self.hovered_completion_index: Optional[int] = None
        self.hovered_target_index: Optional[int] = None
        self.multi_saccade_view = False  # Switches between single-color or multi-phase saccade display
        self.show_scanpath = True
        self.show_fixations = True
        self.show_saccades = False

        # Dropdown controls
        self.dropdown_open = False
        self.dropdown_rect = pygame.Rect(20, self.height - 110, 220, 40)
        self.expanded_dropdown_rect = pygame.Rect(20, 0, 220, 0)
        self.current_dropdown_height = 0.0
        self.dropdown_anim_speed = 15.0

        # Button rectangles
        self.velocity_btn_rect = pygame.Rect(20, self.height - 60, 220, 40)
        self.metrics_btn_rect = pygame.Rect(self.width - 220, 20, 200, 40)
        self.expanded_velocity_rect = pygame.Rect(50, 50, self.width - 400, 150)
        self.expanded_metrics_rect = pygame.Rect(self.width - 350, 200, 330, 420)

        # Timeline management
        self.timeline_btn_rect = pygame.Rect(self.width - 220, 70, 200, 40)
        self.expanded_timeline_rect = pygame.Rect(self.width - 350, 200, 330, 420)
        self.timeline_panel = TimelinePanel(self)

        # Filter panel
        self.filter_panel = FilterPanel(self)
        self.filter_btn_rect = pygame.Rect(self.width - 220, 120, 200, 40)

        # Misc
        self.HOVER_RADIUS = 40
        self.has_saved_image = False
        self.last_draw_data = None

        # For marking target completions and tooltips
        self.completion_marker_positions: List[Tuple[pygame.Rect, int]] = []
        self.target_positions: List[Tuple[int, int]] = []
        self.completion_times: Optional[List[float]] = None

        # Visualization sub-components
        self.gaze_path_vis = GazePathVisualization(self)
        self.fixations_vis = FixationsVisualization(self)
        self.saccades_vis = SaccadesVisualization(self)
        self.targets_vis = TargetsVisualization(self)
        self.velocity_vis = VelocityProfileVisualization(self)
        self.metrics_vis = MetricsSummaryVisualization(self)

    @property
    def saccade_single_mode(self) -> bool:
        """
        Indicates if the saccades are drawn using a single color,
        or if multi-phase rendering is used.
        """
        return not self.multi_saccade_view

    @saccade_single_mode.setter
    def saccade_single_mode(self, val: bool) -> None:
        """
        Toggle between single-color and multi-color (multi_saccade_view) modes.
        """
        if val:
            self.multi_saccade_view = False

    @property
    def saccade_multi_mode(self) -> bool:
        """
        Indicates if saccades are rendered in multiple velocity-based colors (multi_saccade_view).
        """
        return self.multi_saccade_view

    @saccade_multi_mode.setter
    def saccade_multi_mode(self, val: bool) -> None:
        """
        Toggle between multi-saccade (velocity-coded) mode and single-color mode.
        """
        if val:
            self.multi_saccade_view = True

    def draw(self,
             gaze_points: List[GazePoint],
             fixations: List[Fixation],
             saccades: List[Saccade],
             heatmap_data: Dict[Tuple[int, int], int],
             target_positions: List[Tuple[int, int]],
             completion_times: Optional[List[float]] = None,
             completion_times_absolute: Optional[List[float]] = None) -> None:
        """
        Main drawing method for the analysis view. Renders all active visual elements,
        applies filters, and updates UI components like tooltips and toggles.

        Args:
            gaze_points (List[GazePoint]): All raw gaze points recorded.
            fixations (List[Fixation]): Detected fixations from analysis.
            saccades (List[Saccade]): Detected saccades from analysis.
            heatmap_data (Dict[Tuple[int,int],int]): Heatmap intensity data.
            target_positions (List[Tuple[int,int]]): Positions of any targets used.
            completion_times (Optional[List[float]]): Relative completion times for each target.
            completion_times_absolute (Optional[List[float]]): Absolute completion timestamps.
        """
        # Remember the last data set we drew
        self.last_draw_data = (gaze_points, fixations, saccades, heatmap_data,
                               target_positions, completion_times)
        self.target_positions = target_positions
        self.completion_times = completion_times

        # 1) Update filter panel with new data ranges, apply cutoffs
        self.filter_panel.update_data_ranges(fixations, saccades)
        filtered_fixations = [
            f for f in fixations
            if f.duration >= self.filter_panel.get_fixation_cutoff()
        ]
        filtered_saccades = [
            s for s in saccades
            if s.distance_traveled >= self.filter_panel.get_saccade_cutoff()
        ]

        # 2) Clear the screen and draw the background grid
        self.screen.fill(self.BACKGROUND_COLOR)
        self.draw_grid()

        # 3) Draw each visualization if enabled
        if self.show_scanpath:
            self.gaze_path_vis.draw_gaze_path(gaze_points)

        if self.show_saccades:
            self.saccades_vis.draw_saccades(filtered_saccades)
            if self.multi_saccade_view:
                self.saccades_vis.draw_legend()

        if self.show_fixations:
            self.fixations_vis.draw_fixations(filtered_fixations, self.hover_fixation)

        self.targets_vis.draw_targets(target_positions, completion_times,
                                      self.hovered_target_index, self.hovered_completion_index)

        self.velocity_vis.draw_velocity_profile(gaze_points, target_positions,
                                                completion_times, self.velocity_profile_expanded)

        self.metrics_vis.draw_metrics_summary(gaze_points,
                                              filtered_fixations,
                                              target_positions,
                                              self.metrics_expanded,
                                              completion_times)

        # 4) Filter panel rendering
        self.filter_panel.draw(fixations, saccades)

        # 5) Hover detection and tooltips
        mouse_pos = pygame.mouse.get_pos()
        self._handle_hover(mouse_pos, filtered_fixations, filtered_saccades)

        # 6) If something is selected, draw its tooltip
        #    a) Fixation
        if self.show_fixations and self.selected_fixation and self.selected_fixation != self.hover_fixation:
            fxpos = self.selected_fixation.center_position
            tooltip_pos = (fxpos[0], fxpos[1] - 50)
            self.fixations_vis.draw_tooltip(self.selected_fixation, tooltip_pos)

        #    b) Saccade
        if self.show_saccades and self.selected_saccade:
            sx = (self.selected_saccade.start_position[0] + self.selected_saccade.end_position[0]) // 2
            sy = (self.selected_saccade.start_position[1] + self.selected_saccade.end_position[1]) // 2
            self.saccades_vis.draw_tooltip((self.selected_saccade, None), (sx, sy - 50))

        #    c) Target
        if self.selected_target_index is not None:
            if 0 <= self.selected_target_index < len(target_positions):
                pos = target_positions[self.selected_target_index]
                lines = [
                    f"Target #{self.selected_target_index + 1}",
                    (f"Completion: {self.completion_times[self.selected_target_index]:.2f}s"
                     if self.completion_times and
                     self.selected_target_index < len(self.completion_times) else "")
                ]
                self.targets_vis.draw_tooltip(self.selected_target_index, pos, lines)

        # 7) Draw the main UI buttons and dropdown
        self._draw_buttons()
        self._draw_visualization_dropdown()

        # 8) Draw the timeline panel
        self.timeline_panel.draw(filtered_fixations,
                                 filtered_saccades,
                                 completion_times_absolute)

        # 9) Present the final rendered frame
        pygame.display.flip()

    def update(self, dt: float) -> None:
        """
        Periodic update call to handle animations (dropdown expansion).
        
        Args:
            dt (float): Time elapsed since the last update in seconds.
        """
        target_h = 0
        if self.dropdown_open:
            # Base items in the dropdown
            base_items = 3
            # If saccade is enabled, show extra items for single/multi mode
            if self.show_saccades:
                base_items += 2
            target_h = base_items * 40

        # Animate opening or closing of the dropdown
        if self.current_dropdown_height < target_h:
            self.current_dropdown_height = min(self.current_dropdown_height + self.dropdown_anim_speed,
                                               float(target_h))
        elif self.current_dropdown_height > target_h:
            self.current_dropdown_height = max(self.current_dropdown_height - self.dropdown_anim_speed,
                                               0.0)

    def handle_event(self, event: pygame.event.Event) -> None:
        """
        Handle input events (mouse, keyboard). Interprets clicks on the dropdown,
        filter panel, timeline panel, or other UI components.

        Args:
            event (pygame.event.Event): The Pygame event to process.
        """
        mouse_pos = pygame.mouse.get_pos()

        # 1) Handle filter panel events first if open
        if self.filter_panel.open:
            self.filter_panel.handle_event(event)
            if event.type == pygame.MOUSEBUTTONDOWN:
                # If clicking outside the panel, close it
                if not self.filter_panel.panel_rect.collidepoint(mouse_pos):
                    self.filter_panel.open = False
                return

        # 2) Dropdown toggling
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Toggle dropdown
            if self.dropdown_rect.collidepoint(mouse_pos):
                self.dropdown_open = not self.dropdown_open
                return

            # If dropdown is open, handle clicks on items
            if self.dropdown_open:
                expanded_x = self.dropdown_rect.x
                expanded_y = self.dropdown_rect.y - self.current_dropdown_height
                expanded_area = pygame.Rect(expanded_x,
                                            expanded_y,
                                            self.dropdown_rect.width,
                                            int(self.current_dropdown_height))
                if expanded_area.collidepoint(mouse_pos):
                    # Determine which menu item was clicked
                    local_y = mouse_pos[1] - expanded_y
                    idx = int(local_y // 40)
                    entries = [
                        ("Scanpath", "show_scanpath", False),
                        ("Fixations", "show_fixations", False),
                        ("Saccades", "show_saccades", False),
                    ]
                    # If saccade display is on, we add additional toggles
                    if self.show_saccades:
                        entries.append(("Saccade Single Color", "saccade_single_mode", True))
                        entries.append(("Saccade Multi-Phase", "saccade_multi_mode", True))
                    if 0 <= idx < len(entries):
                        _, attr, _ = entries[idx]
                        cur = getattr(self, attr)
                        setattr(self, attr, not cur)
                    return
                else:
                    # Clicked outside => close
                    self.dropdown_open = False

            # 3) Other UI buttons
            if self.velocity_btn_rect.collidepoint(mouse_pos):
                self.velocity_profile_expanded = not self.velocity_profile_expanded
                # Close other panels if velocity is expanded
                if self.velocity_profile_expanded:
                    self.metrics_expanded = False
                    self.timeline_panel.open = False
                    self.filter_panel.open = False

            elif self.metrics_btn_rect.collidepoint(mouse_pos):
                self.metrics_expanded = not self.metrics_expanded
                # Close other panels if metrics is expanded
                if self.metrics_expanded:
                    self.velocity_profile_expanded = False
                    self.timeline_panel.open = False
                    self.filter_panel.open = False

            elif self.timeline_btn_rect.collidepoint(mouse_pos):
                self.timeline_panel.open = not self.timeline_panel.open
                # Close other panels if timeline is expanded
                if self.timeline_panel.open:
                    self.velocity_profile_expanded = False
                    self.metrics_expanded = False
                    self.filter_panel.open = False

            elif self.filter_btn_rect.collidepoint(mouse_pos):
                self.filter_panel.open = not self.filter_panel.open
                # Close other panels if filter is expanded
                if self.filter_panel.open:
                    self.velocity_profile_expanded = False
                    self.metrics_expanded = False
                    self.timeline_panel.open = False
            else:
                # If clicked somewhere else => clear any selected items
                self.selected_target_index = None
                self.selected_fixation = None
                self.selected_saccade = None

        # 4) If timeline panel is open, pass the event
        if self.timeline_panel.open and self.last_draw_data is not None:
            _, fixations, saccades, _, _, _ = self.last_draw_data
            self.timeline_panel.handle_event(event)

    def save_analysis_image(self, suffix: str = "") -> None:
        """
        Saves a screenshot of the current analysis view.

        Args:
            suffix (str): Optional string to append to the filename.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"gaze_analysis{suffix}_{ts}.png"
        pygame.image.save(self.screen, fname)

    def draw_grid(self) -> None:
        """
        Draws a grid across the background to help situate the user’s eye data visually.
        """
        for x in range(0, self.width, 50):
            pygame.draw.line(self.screen, self.GRID_COLOR, (x, 0), (x, self.height))
        for y in range(0, self.height, 50):
            pygame.draw.line(self.screen, self.GRID_COLOR, (0, y), (self.width, y))

    def _handle_hover(self,
                      mouse_pos: Tuple[int, int],
                      fixations: List[Fixation],
                      saccades: List[Saccade]) -> None:
        """
        Detect whether the user is hovering over a fixation, saccade, or target,
        and if so, display the appropriate tooltip.

        Args:
            mouse_pos (Tuple[int,int]): Current mouse coordinates.
            fixations (List[Fixation]): List of displayed fixations.
            saccades (List[Saccade]): List of displayed saccades.
        """
        self.hover_fixation = None
        self.hovered_completion_index = None
        self.hovered_target_index = None

        # If velocity profile is expanded, check if user hovers over completion markers
        if self.velocity_profile_expanded:
            for marker_rect, idx in self.completion_marker_positions:
                if marker_rect.collidepoint(mouse_pos):
                    self.hovered_completion_index = idx
                    return

        # Check if hovering any target
        for i, pos in enumerate(self.target_positions):
            dx = mouse_pos[0] - pos[0]
            dy = mouse_pos[1] - pos[1]
            if dx * dx + dy * dy <= self.HOVER_RADIUS ** 2:
                self.hovered_target_index = i
                # Show basic target tooltip
                self.targets_vis.draw_tooltip(
                    i,
                    pos,
                    [
                        f"Target #{i + 1}",
                        (f"Completion: {self.completion_times[i]:.2f}s"
                         if self.completion_times and i < len(self.completion_times)
                         else "")
                    ]
                )
                return

        # Check saccades if visible
        if self.show_saccades:
            hovered = self.saccades_vis.check_hover(mouse_pos)
            if hovered:
                self.saccades_vis.draw_tooltip(hovered, mouse_pos)
                return

        # Check fixations if visible
        if self.show_fixations:
            for fixation in fixations:
                dx = mouse_pos[0] - fixation.center_position[0]
                dy = mouse_pos[1] - fixation.center_position[1]
                if dx * dx + dy * dy <= self.HOVER_RADIUS ** 2:
                    self.hover_fixation = fixation
                    self.fixations_vis.draw_tooltip(fixation, mouse_pos)
                    return

    def _draw_clean_text(self, text: str, x: int, y: int) -> None:
        """
        Helper to draw text with the default font and color,
        without a background rectangle.

        Args:
            text (str): The string to render.
            x (int): X screen coordinate.
            y (int): Y screen coordinate.
        """
        sf = self.FONT.render(text, True, self.TEXT_COLOR)
        self.screen.blit(sf, (x, y))

    def _draw_buttons(self) -> None:
        """
        Render main UI control buttons along the bottom and top right corners:
        Velocity profile, Metrics, Timeline, and Filters.
        """
        # Velocity profile button
        pygame.draw.rect(self.screen, self.BUTTON_COLOR, self.velocity_btn_rect)
        pygame.draw.rect(self.screen, self.BUTTON_BORDER_COLOR, self.velocity_btn_rect, 2)
        self._draw_clean_text("Velocity Profile",
                              self.velocity_btn_rect.left + 10,
                              self.velocity_btn_rect.top + 10)

        # Metrics button
        pygame.draw.rect(self.screen, self.BUTTON_COLOR, self.metrics_btn_rect)
        pygame.draw.rect(self.screen, self.BUTTON_BORDER_COLOR, self.metrics_btn_rect, 2)
        self._draw_clean_text("Metrics",
                              self.metrics_btn_rect.left + 10,
                              self.metrics_btn_rect.top + 10)

        # Timeline button
        pygame.draw.rect(self.screen, self.BUTTON_COLOR, self.timeline_btn_rect)
        pygame.draw.rect(self.screen, self.BUTTON_BORDER_COLOR, self.timeline_btn_rect, 2)
        self._draw_clean_text("Timeline",
                              self.timeline_btn_rect.left + 10,
                              self.timeline_btn_rect.top + 10)

        # Filter button
        pygame.draw.rect(self.screen, self.BUTTON_COLOR, self.filter_btn_rect)
        pygame.draw.rect(self.screen, self.BUTTON_BORDER_COLOR, self.filter_btn_rect, 2)
        self._draw_clean_text("Filters",
                              self.filter_btn_rect.left + 10,
                              self.filter_btn_rect.top + 10)

    def _draw_visualization_dropdown(self) -> None:
        """
        Draw the dropdown button for toggling visual layers (Scanpath, Fixations, Saccades),
        plus additional toggles if Saccades are on (single-color vs. multi-phase).
        """
        pygame.draw.rect(self.screen, self.BUTTON_COLOR, self.dropdown_rect)
        pygame.draw.rect(self.screen, self.BUTTON_BORDER_COLOR, self.dropdown_rect, 2)
        self._draw_clean_text("Visualization Options",
                              self.dropdown_rect.left + 10,
                              self.dropdown_rect.top + 10)

        # If dropdown is partially or fully open
        if self.current_dropdown_height > 1:
            ex = pygame.Rect(
                self.dropdown_rect.x,
                self.dropdown_rect.y - self.current_dropdown_height,
                self.dropdown_rect.width,
                self.current_dropdown_height
            )
            pygame.draw.rect(self.screen, self.PANEL_COLOR, ex)
            pygame.draw.rect(self.screen, self.BUTTON_BORDER_COLOR, ex, 2)

            # Base menu items
            dd = [
                ("Scanpath", "show_scanpath", False),
                ("Fixations", "show_fixations", False),
                ("Saccades", "show_saccades", False),
            ]
            # If saccade is active, add toggles for single vs. multi-phase
            if self.show_saccades:
                dd.append(("Saccade Single Color", "saccade_single_mode", True))
                dd.append(("Saccade Multi-Phase", "saccade_multi_mode", True))

            # Limit items if the dropdown is still expanding
            max_items = int(self.current_dropdown_height // 40)
            for i, (label, attr, sub) in enumerate(dd[:max_items]):
                iy = ex.top + i * 40
                irect = pygame.Rect(ex.left, iy, ex.width, 40)

                # Highlight if currently active
                val = getattr(self, attr)
                if val:
                    pygame.draw.rect(self.screen, (60, 80, 100), irect)

                # Sub-items are indented further
                indent = 30 if sub else 10
                self._draw_clean_text(label, irect.left + indent, irect.top + 10)