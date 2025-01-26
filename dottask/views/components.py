#views/components.py


# =========================================================================================
# COMPONENTS.PY
#
# Contains a collection of visualization classes and supporting data structures used
# to display gaze paths, saccades, fixations, targets, velocity profiles, and metrics.
# These classes integrate with the parent view to render on-screen elements and handle
# interaction such as hover detection.
# =========================================================================================




import pygame
import pygame.gfxdraw
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Union

@dataclass
class GazePoint:
    """
    Represents a single gaze point measurement.

    Attributes:
        timestamp (float): The time when this gaze point was recorded.
        position (Tuple[int, int]): The (x, y) screen coordinates of the gaze.
        velocity (float): Estimated velocity (px/s) at this point.
    """
    timestamp: float
    position: Tuple[int, int]
    velocity: float

@dataclass
class Fixation:
    """
    Represents a period of stable gaze.

    Attributes:
        start_time (float): Analysis-relative or derived start time.
        end_time (float): Analysis-relative or derived end time.
        center_position (Tuple[int,int]): The (x, y) centroid of all included points.
        duration (float): Duration of this fixation in seconds.
        gaze_points (List[GazePoint]): The raw gaze data points encompassed by this fixation.
        start_time_absolut (float): Absolute start timestamp (from raw data).
        end_time_absolut (float): Absolute end timestamp (from raw data).
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
        start_position (Tuple[int,int]): Segment start position on-screen.
        end_position (Tuple[int,int]): Segment end position on-screen.
        mean_velocity (float): Mean velocity (px/s) within this segment.
        peak_velocity (float): Peak velocity (px/s) within this segment.
        gaze_point_count (int): Number of gaze points in this sub-segment.
    """
    start_position: Tuple[int, int]
    end_position: Tuple[int, int]
    mean_velocity: float
    peak_velocity: float
    gaze_point_count: int

@dataclass
class Saccade:
    """
    Represents rapid eye movement between fixations.

    Attributes:
        start_time (float): Analysis-relative start time of this saccade.
        end_time (float): Analysis-relative end time of this saccade.
        start_time_absolut (float): Absolute start timestamp (from raw data).
        end_time_absolut (float): Absolute end timestamp (from raw data).
        start_position (Tuple[int,int]): (x, y) start of the saccade.
        end_position (Tuple[int,int]): (x, y) end of the saccade.
        duration (float): Saccade duration in seconds.
        peak_velocity (float): Maximum velocity reached in this saccade (px/s).
        mean_velocity (float): Average velocity of the entire saccade (px/s).
        amplitude (float): Straight-line distance between start and end positions.
        distance_traveled (float): Sum of point-to-point distances (path length).
        gaze_points (List[GazePoint]): All gaze points forming the saccade.
        segments (Optional[List[SaccadeSegment]]): Velocity-based sub-segments (None if single-phase).
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
    segments: Optional[List[SaccadeSegment]] = None  # None => single-phase


# --------------------------------------------------------------------------------------
# Visualization Classes
# --------------------------------------------------------------------------------------
class GazePathVisualization:
    """
    Displays the raw gaze path as a continuous series of anti-aliased lines.
    """

    def __init__(self, parent):
        """
        Args:
            parent: Reference to the AnalysisView or another view that uses this component.
        """
        self.parent = parent

    def draw_gaze_path(self, gaze_points: List[GazePoint]) -> None:
        """
        Render a line path connecting all provided gaze points in chronological order.

        Args:
            gaze_points (List[GazePoint]): Raw gaze points to connect.
        """
        if len(gaze_points) > 1:
            points = [(p.position[0], p.position[1]) for p in gaze_points]
            pygame.draw.aalines(
                self.parent.screen,
                self.parent.CHART_COLOR,
                False,
                points,
                blend=1
            )


class SaccadesVisualization:
    """
    Visualizes saccades on-screen with optional velocity-based coloring.
    Includes arrow markings to indicate direction and start/end highlights.
    """

    def __init__(self, parent):
        """
        Args:
            parent: Reference to the main view or analysis class.
        """
        self.parent = parent
        self.SACCADE_COLOR = (255, 0, 200)  # Distinct default color for saccades

        # For hover detection
        self.saccade_segments: List[Tuple[pygame.Rect, Saccade, Optional[SaccadeSegment]]] = []

    def draw_saccades(self, saccades: List[Saccade]) -> None:
        """
        Render each saccade as a line with optional velocity-based segmentation.

        Args:
            saccades (List[Saccade]): List of saccades to be drawn.
        """
        self.saccade_segments.clear()
        for saccade in saccades:
            start = saccade.start_position
            end = saccade.end_position

            # Multi-phase color-coding if segments are present and multi_saccade_view is enabled
            if self.parent.multi_saccade_view and saccade.segments:
                segments = saccade.segments
                velocities = [seg.mean_velocity for seg in segments]
                min_vel = min(velocities) if velocities else 0.0
                max_vel = max(velocities) if velocities else 1.0

                for seg in segments:
                    seg_color = self._velocity_to_color(seg.mean_velocity, min_vel, max_vel)
                    pygame.draw.aaline(
                        self.parent.screen,
                        seg_color,
                        seg.start_position,
                        seg.end_position,
                        1
                    )

                    # Store bounding rect for hover checks
                    r_x = min(seg.start_position[0], seg.end_position[0]) - 5
                    r_y = min(seg.start_position[1], seg.end_position[1]) - 5
                    r_w = abs(seg.end_position[0] - seg.start_position[0]) + 10
                    r_h = abs(seg.end_position[1] - seg.start_position[1]) + 10
                    self.saccade_segments.append((pygame.Rect(r_x, r_y, r_w, r_h), saccade, seg))
            else:
                # Single-phase saccade in a solid color
                pygame.draw.aaline(self.parent.screen, self.SACCADE_COLOR, start, end, 1)
                thickness = 10
                rect_x = min(start[0], end[0]) - thickness
                rect_y = min(start[1], end[1]) - thickness
                rect_w = abs(end[0] - start[0]) + 2 * thickness
                rect_h = abs(end[1] - start[1]) + 2 * thickness
                self.saccade_segments.append((pygame.Rect(rect_x, rect_y, rect_w, rect_h), saccade, None))

            # Mark start and draw direction arrow
            self._draw_saccade_arrows(saccade, start, end)

    def check_hover(self, mouse_pos: Tuple[int, int]) -> Optional[Tuple[Saccade, Optional[SaccadeSegment]]]:
        """
        Determines if the mouse is hovering over a saccade or its segment bounding box.
        If so, calculates the distance to the line to confirm an actual hover.

        Args:
            mouse_pos (Tuple[int,int]): Current mouse coordinates.

        Returns:
            Optional[Tuple[Saccade, Optional[SaccadeSegment]]]: The hovered saccade and segment
            (None if not hovering over any).
        """
        for rect, saccade, segment in self.saccade_segments:
            if rect.collidepoint(mouse_pos):
                if segment:
                    dist = self._point_line_distance(mouse_pos, segment.start_position, segment.end_position)
                else:
                    dist = self._point_line_distance(mouse_pos, saccade.start_position, saccade.end_position)
                if dist < 10:
                    return (saccade, segment)
        return None

    def draw_tooltip(self, hovered_data: Tuple[Saccade, Optional[SaccadeSegment]], mouse_pos: Tuple[int, int]) -> None:
        """
        Render a tooltip showing saccade or saccade-segment metrics at the mouse position.

        Args:
            hovered_data (Tuple[Saccade, Optional[SaccadeSegment]]): The hovered saccade
                and an optional sub-segment.
            mouse_pos (Tuple[int,int]): Coordinates at which to draw the tooltip.
        """
        saccade, seg = hovered_data
        if seg is None:
            # Tooltip for entire saccade
            lines = [
                f"Duration: {saccade.duration:.2f}s",
                f"Distance: {saccade.distance_traveled:.2f}px",
                f"Mean Velocity: {saccade.mean_velocity:.2f}px/s",
                f"Peak Velocity: {saccade.peak_velocity:.2f}px/s",
                f"Amplitude: {saccade.amplitude:.2f}px",
                f"Gaze Points: {len(saccade.gaze_points)}"
            ]
        else:
            # Tooltip for a velocity sub-segment
            lines = [
                "Saccade Segment",
                f"Mean Velocity: {seg.mean_velocity:.2f}px/s",
                f"Peak Velocity: {seg.peak_velocity:.2f}px/s",
                f"Gaze Points: {seg.gaze_point_count}"
            ]

        # Build text surfaces
        surfaces = []
        max_w = 0
        for txt in lines:
            surf = self.parent.FONT.render(txt, True, self.parent.TEXT_COLOR)
            surfaces.append(surf)
            max_w = max(max_w, surf.get_width())

        line_h = self.parent.FONT.get_height()
        total_h = len(surfaces) * (line_h + 5) + 10

        # Position the tooltip based on mouse location
        if mouse_pos[0] < self.parent.width / 2:
            x = mouse_pos[0] + 20
        else:
            x = mouse_pos[0] - (max_w + 10) - 20
        y = mouse_pos[1] - total_h / 2

        tooltip_rect = pygame.Rect(x, y, max_w + 10, total_h)
        pygame.draw.rect(self.parent.screen, self.parent.TOOLTIP_COLOR, tooltip_rect)
        pygame.draw.rect(self.parent.screen, self.parent.GRID_COLOR, tooltip_rect, 1)

        # Blit text lines
        cur_y = y + 5
        for s in surfaces:
            self.parent.screen.blit(s, (x + 5, cur_y))
            cur_y += line_h + 5

    def draw_legend(self) -> None:
        """
        Draw a simple legend indicating velocity colors for multi-saccade mode.
        """
        legend_w = 150
        legend_h = 60
        x = self.parent.width - legend_w - 30
        y = self.parent.height - legend_h - 20
        rect = pygame.Rect(x, y, legend_w, legend_h)

        pygame.draw.rect(self.parent.screen, self.parent.PANEL_COLOR, rect)
        pygame.draw.rect(self.parent.screen, self.parent.GRID_COLOR, rect, 1)

        label = self.parent.FONT.render("Mean Velocity", True, self.parent.TEXT_COLOR)
        lx = rect.centerx - (label.get_width() // 2)
        ly = rect.top + 5
        self.parent.screen.blit(label, (lx, ly))

        margin = 10
        bar_top = ly + label.get_height() + 10
        bar_h = 10
        center_y = bar_top + bar_h // 2

        # Low/High velocity labels
        low_surf = self.parent.FONT.render("Low", True, self.parent.TEXT_COLOR)
        high_surf = self.parent.FONT.render("High", True, self.parent.TEXT_COLOR)
        low_pos = (rect.left + margin, center_y - low_surf.get_height() // 2)
        high_pos = (rect.right - margin - high_surf.get_width(),
                    center_y - high_surf.get_height() // 2)
        self.parent.screen.blit(low_surf, low_pos)
        self.parent.screen.blit(high_surf, high_pos)

        # Gradient bar
        grad_left = low_pos[0] + low_surf.get_width() + margin
        grad_right = high_pos[0] - margin
        grad_w = grad_right - grad_left
        if grad_w <= 0:
            return
        min_clr = (0, 255, 0)   # Low velocity color
        max_clr = (255, 0, 0)   # High velocity color
        for i in range(grad_w):
            ratio = i / (grad_w - 1)
            rr = int(min_clr[0] + ratio * (max_clr[0] - min_clr[0]))
            gg = int(min_clr[1] + ratio * (max_clr[1] - min_clr[1]))
            bb = int(min_clr[2] + ratio * (max_clr[2] - min_clr[2]))
            pygame.draw.line(self.parent.screen, (rr, gg, bb),
                             (grad_left + i, bar_top),
                             (grad_left + i, bar_top + bar_h))

    # ------------------------------------------------------------------------
    # Internal Helper Methods
    # ------------------------------------------------------------------------
    def _velocity_to_color(self, v: float, mn: float, mx: float) -> Tuple[int, int, int]:
        """
        Convert a velocity value to a color scale between two extremes.

        Args:
            v (float): The velocity value to map.
            mn (float): Minimum velocity across all segments.
            mx (float): Maximum velocity across all segments.

        Returns:
            (r, g, b) color tuple representing the velocity on a gradient.
        """
        if mn == mx:
            return (128, 128, 0)  # Default color if no range
        ratio = (v - mn) / (mx - mn)
        # Shift from olive green (128,128,0) to red (255,0,0)
        r = int(128 + 127 * ratio)
        g = int(128 - 128 * ratio)
        return (r, g, 0)

    def _draw_saccade_arrows(self, saccade: Saccade, start: Tuple[int, int], end: Tuple[int, int]) -> None:
        """
        Draw markers at the start of a saccade and an arrowhead at the end.

        Args:
            saccade (Saccade): The saccade being drawn.
            start (Tuple[int,int]): Saccade start coords.
            end (Tuple[int,int]): Saccade end coords.
        """
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.hypot(dx, dy)
        if length < 1e-9:
            return

        dx /= length
        dy /= length
        perp_dx = -dy
        perp_dy = dx

        # Start marker
        start_line_len = 6
        line_left = (start[0] + perp_dx * (start_line_len / 2),
                     start[1] + perp_dy * (start_line_len / 2))
        line_right = (start[0] - perp_dx * (start_line_len / 2),
                      start[1] - perp_dy * (start_line_len / 2))

        start_color = self.SACCADE_COLOR
        if self.parent.multi_saccade_view and saccade.segments:
            first_seg = saccade.segments[0]
            seg_min = min(seg.mean_velocity for seg in saccade.segments)
            seg_max = max(seg.mean_velocity for seg in saccade.segments)
            start_color = self._velocity_to_color(first_seg.mean_velocity, seg_min, seg_max)

        pygame.draw.aaline(self.parent.screen, start_color, line_left, line_right, 1)

        # Arrow tip
        arrow_color = start_color
        if self.parent.multi_saccade_view and saccade.segments:
            last_seg = saccade.segments[-1]
            seg_min = min(seg.mean_velocity for seg in saccade.segments)
            seg_max = max(seg.mean_velocity for seg in saccade.segments)
            arrow_color = self._velocity_to_color(last_seg.mean_velocity, seg_min, seg_max)

        arrow_size = 10
        arrow_x = end[0] - dx * arrow_size
        arrow_y = end[1] - dy * arrow_size
        arrow_points = [
            end,
            (arrow_x + perp_dx * arrow_size / 2, arrow_y + perp_dy * arrow_size / 2),
            (arrow_x - perp_dx * arrow_size / 2, arrow_y - perp_dy * arrow_size / 2)
        ]
        self._draw_aa_polygon(self.parent.screen, arrow_points, arrow_color)

    def _draw_aa_polygon(self, surf: pygame.Surface, pts: List[Tuple[float, float]], clr: Tuple[int, int, int]) -> None:
        """
        Draws an anti-aliased filled polygon.

        Args:
            surf (pygame.Surface): Surface to draw on.
            pts (List[Tuple[float,float]]): Polygon vertices.
            clr (Tuple[int,int,int]): (r, g, b) color tuple.
        """
        ipts = [(int(px), int(py)) for px, py in pts]
        pygame.gfxdraw.aapolygon(surf, ipts, clr)
        pygame.gfxdraw.filled_polygon(surf, ipts, clr)

    def _point_line_distance(self, pt: Tuple[int, int], a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """
        Calculate the minimum distance from a point to a line segment.

        Args:
            pt (Tuple[int,int]): The point (mouse position).
            a (Tuple[int,int]): Start of the line segment.
            b (Tuple[int,int]): End of the line segment.

        Returns:
            float: Distance in pixels from the point to the line.
        """
        px, py = pt
        x1, y1 = a
        x2, y2 = b
        lensq = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if lensq < 1e-9:
            return np.hypot(px - x1, py - y1)
        t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / lensq
        t = max(0, min(1, t))
        projx = x1 + t * (x2 - x1)
        projy = y1 + t * (y2 - y1)
        return np.hypot(px - projx, py - projy)


class FixationsVisualization:
    """
    Visualizes detected fixations as circles around their center positions.
    """

    def __init__(self, parent):
        """
        Args:
            parent: Reference to the parent view or analysis class.
        """
        self.parent = parent

    def draw_fixations(self, fixations: List[Fixation], hover_fixation: Optional[Fixation]) -> None:
        """
        Draw each fixation as a circle around its center, highlighting if hovered or selected.

        Args:
            fixations (List[Fixation]): List of fixations to draw.
            hover_fixation (Optional[Fixation]): The fixation currently under mouse hover, if any.
        """
        surf = pygame.Surface((self.parent.width, self.parent.height), pygame.SRCALPHA)
        for f in fixations:
            # Increase alpha if hovered or selected
            if f == hover_fixation or f == self.parent.selected_fixation:
                alpha = 200
            else:
                alpha = 128
            self.draw_aa_circle(surf, f.center_position, 8, (*self.parent.FIXATION_COLOR, alpha))
        self.parent.screen.blit(surf, (0, 0))

    def draw_tooltip(self, fixation: Fixation, mouse_pos: Tuple[int, int]) -> None:
        """
        Display a tooltip for the given fixation at the specified mouse position.

        Args:
            fixation (Fixation): The fixation whose info we want to display.
            mouse_pos (Tuple[int,int]): Tooltip display coordinates.
        """
        lines = [
            f"Duration: {fixation.duration:.2f}s",
            f"Points: {len(fixation.gaze_points)}"
        ]
        surfaces = []
        max_w = 0
        for ln in lines:
            surf = self.parent.FONT.render(ln, True, self.parent.TEXT_COLOR)
            surfaces.append(surf)
            max_w = max(max_w, surf.get_width())

        line_h = self.parent.FONT.get_height()
        total_h = len(surfaces) * (line_h + 5) + 10
        if mouse_pos[0] < self.parent.width / 2:
            x = mouse_pos[0] + 20
        else:
            x = mouse_pos[0] - (max_w + 10) - 20
        y = mouse_pos[1] - total_h / 2

        rect = pygame.Rect(x, y, max_w + 10, total_h)
        pygame.draw.rect(self.parent.screen, self.parent.TOOLTIP_COLOR, rect)
        pygame.draw.rect(self.parent.screen, self.parent.GRID_COLOR, rect, 1)

        cur_y = y + 5
        for sf in surfaces:
            self.parent.screen.blit(sf, (x + 5, cur_y))
            cur_y += line_h + 5

    def draw_aa_circle(self, surface, center, radius, color):
        """
        Safely draw an anti-aliased circle. If gfxdraw is not available,
        fallback to the standard Pygame circle.

        Args:
            surface (pygame.Surface): Surface to draw on.
            center (Tuple[int,int]): The (x, y) center of the circle.
            radius (int): Circle radius in pixels.
            color (Tuple[int,int,int,int]): RGBA color.
        """
        try:
            pygame.gfxdraw.aacircle(surface, center[0], center[1], radius, color)
            pygame.gfxdraw.filled_circle(surface, center[0], center[1], radius, color)
        except ImportError:
            pygame.draw.circle(surface, color, center, radius)


class TargetsVisualization:
    """
    Draws target markers (e.g., for verification or training tasks) and their tooltips.
    """

    def __init__(self, parent):
        """
        Args:
            parent: Reference to the parent view object.
        """
        self.parent = parent

    def draw_targets(self,
                     target_positions: List[Tuple[int, int]],
                     completion_times: Optional[List[float]],
                     hovered_target_index: Optional[int],
                     hovered_completion_index: Optional[int]) -> None:
        """
        Draw each target dot, highlighting any hovered or selected targets.

        Args:
            target_positions (List[Tuple[int,int]]): List of (x, y) positions for targets.
            completion_times (Optional[List[float]]): Relative completion times for each target.
            hovered_target_index (Optional[int]): Which target index is hovered, if any.
            hovered_completion_index (Optional[int]): Which target index is hovered in velocity panel, if any.
        """
        for i, pos in enumerate(target_positions):
            is_hovered = (i == hovered_target_index or i == hovered_completion_index)
            is_selected = (i == self.parent.selected_target_index)
            if is_hovered or is_selected:
                # Draw an emphasized target
                clr = self.parent.TARGET_COLOR
                pygame.draw.circle(self.parent.screen, clr, pos, 15, 2)
                pygame.draw.circle(self.parent.screen, clr, pos, 10, 2)
                pygame.draw.circle(self.parent.screen, clr, pos, 3)
                lines = [f"Target #{i + 1}"]
                if completion_times and i < len(completion_times):
                    lines.append(f"Completion: {completion_times[i]:.2f}s")
                self.draw_tooltip(i, pos, lines)
            else:
                # Faded style if not hovered or selected
                clr = (*self.parent.TARGET_COLOR, 128)
                su = pygame.Surface((40, 40), pygame.SRCALPHA)
                pygame.draw.circle(su, clr, (20, 20), 15, 2)
                pygame.draw.circle(su, clr, (20, 20), 10, 2)
                pygame.draw.circle(su, clr, (20, 20), 3)
                self.parent.screen.blit(su, (pos[0] - 20, pos[1] - 20))

    def draw_tooltip(self, target_index: int, pos: Tuple[int, int], lines: List[str]) -> None:
        """
        Render a tooltip showing target-related info.

        Args:
            target_index (int): Index of the target within the overall target list.
            pos (Tuple[int,int]): The (x, y) coordinates of the target on-screen.
            lines (List[str]): Lines of text to display in the tooltip.
        """
        surfaces = []
        max_w = 0
        for txt in lines:
            surf = self.parent.FONT.render(txt, True, self.parent.TEXT_COLOR)
            surfaces.append(surf)
            max_w = max(max_w, surf.get_width())

        line_h = self.parent.FONT.get_height()
        total_h = len(surfaces) * (line_h + 5) + 10

        # Decide tooltip anchor (left or right of the target)
        if pos[0] < self.parent.width / 2:
            x = pos[0] + 20
        else:
            x = pos[0] - (max_w + 10) - 20
        y = pos[1] - total_h / 2

        tooltip_rect = pygame.Rect(x, y, max_w + 10, total_h)
        pygame.draw.rect(self.parent.screen, self.parent.TOOLTIP_COLOR, tooltip_rect)
        pygame.draw.rect(self.parent.screen, self.parent.GRID_COLOR, tooltip_rect, 1)

        cur_y = y + 5
        for surf in surfaces:
            self.parent.screen.blit(surf, (x + 5, cur_y))
            cur_y += line_h + 5


class VelocityProfileVisualization:
    """
    Visualizes gaze velocity over time, potentially with target completion markers.
    """

    def __init__(self, parent):
        """
        Args:
            parent: The parent view or analysis object.
        """
        self.parent = parent

    def draw_velocity_profile(self,
                              gaze_points: List[GazePoint],
                              target_positions: List[Tuple[int, int]],
                              completion_times: Optional[List[float]],
                              velocity_profile_expanded: bool) -> None:
        """
        Render a velocity vs. time chart in an expanded panel.

        Args:
            gaze_points (List[GazePoint]): The gaze points to extract velocity/time from.
            target_positions (List[Tuple[int,int]]): Not directly used, but kept for consistency.
            completion_times (Optional[List[float]]): Times at which targets were completed.
            velocity_profile_expanded (bool): Whether the velocity panel is open/visible.
        """
        if velocity_profile_expanded and len(gaze_points) >= 2:
            rect = self.parent.expanded_velocity_rect
            # Draw panel background
            pygame.draw.rect(self.parent.screen, self.parent.PANEL_COLOR, rect)
            pygame.draw.rect(self.parent.screen, self.parent.GRID_COLOR, rect, 1)

            vs = [p.velocity for p in gaze_points]
            ts = [p.timestamp - gaze_points[0].timestamp for p in gaze_points]
            max_t = ts[-1] if ts else 1.0
            max_v = max(vs) if vs else 1.0

            left = rect.left + 100
            right = rect.right - 10
            top = rect.top + 10
            bottom = rect.bottom - 30
            axis_y = bottom

            # Axes
            pygame.draw.line(self.parent.screen, self.parent.GRID_COLOR, (left, axis_y), (right, axis_y))
            pygame.draw.line(self.parent.screen, self.parent.GRID_COLOR, (left, top), (left, axis_y))

            # Y-axis labels (velocity)
            steps = 5
            usable_h = axis_y - top
            for i in range(steps + 1):
                ratio = i / steps
                vv = ratio * max_v
                y = axis_y - ratio * usable_h
                lbl = f"{int(vv):,} px/s"
                self.parent._draw_clean_text(lbl, rect.left + 10, y - 10)

            # Plot velocity line
            pw = right - left
            pts = []
            for i in range(len(vs)):
                t = ts[i]
                v = vs[i]
                if max_t > 0:
                    x = int(left + (t / max_t) * pw)
                else:
                    x = left
                if max_v > 0:
                    yy = int(axis_y - (v / max_v) * usable_h)
                else:
                    yy = axis_y
                pts.append((x, yy))

            if len(pts) > 1:
                pygame.draw.aalines(self.parent.screen, self.parent.CHART_COLOR, False, pts, blend=1)

            # Draw target completion markers
            self.parent.completion_marker_positions.clear()
            if completion_times:
                for i, ctime in enumerate(completion_times):
                    if max_t > 0:
                        mx = int(left + (ctime / max_t) * pw)
                    else:
                        mx = left
                    rect2 = pygame.Rect(mx - 10, top, 20, bottom - top)
                    self.parent.completion_marker_positions.append((rect2, i))
                    col = (self.parent.TARGET_COLOR if i == self.parent.hovered_completion_index
                           else (*self.parent.TARGET_COLOR, 128))
                    pygame.draw.line(self.parent.screen, col, (mx, top), (mx, bottom), 2)

            # X-axis time markers
            if max_t > 0:
                step = 2
                for sec in range(0, int(max_t) + 1, step):
                    xx = left + (sec / max_t) * pw
                    self.parent._draw_clean_text(f"{sec}s", int(xx) - 10, axis_y + 5)


class MetricsSummaryVisualization:
    """
    Displays a panel with various computed metrics such as total path length,
    average velocity, fixation durations, and target completion times.
    """

    def __init__(self, parent):
        """
        Args:
            parent: Reference to the parent view or analysis object.
        """
        self.parent = parent

    def draw_metrics_summary(self,
                             gaze_points: List[GazePoint],
                             fixations: List[Fixation],
                             target_positions: List[Tuple[int, int]],
                             metrics_expanded: bool,
                             completion_times: Optional[List[float]]) -> None:
        """
        Draw a textual summary of computed metrics if the metrics panel is expanded.

        Args:
            gaze_points (List[GazePoint]): Raw gaze points for velocity/time range.
            fixations (List[Fixation]): Detected fixations for average durations and counts.
            target_positions (List[Tuple[int,int]]): List of target positions (for completeness).
            metrics_expanded (bool): Whether to show the metrics panel.
            completion_times (Optional[List[float]]): Times at which targets were completed.
        """
        if metrics_expanded and gaze_points:
            r = self.parent.expanded_metrics_rect
            pygame.draw.rect(self.parent.screen, self.parent.PANEL_COLOR, r)
            pygame.draw.rect(self.parent.screen, self.parent.GRID_COLOR, r, 1)

            # Compute basic metrics
            scan_dur = 0.0
            if len(gaze_points) > 1:
                scan_dur = gaze_points[-1].timestamp - gaze_points[0].timestamp

            vs = [p.velocity for p in gaze_points]
            avg_v = sum(vs) / len(vs) if vs else 0
            max_v = max(vs) if vs else 0

            dist = 0.0
            for p1, p2 in zip(gaze_points[:-1], gaze_points[1:]):
                dx = p2.position[0] - p1.position[0]
                dy = p2.position[1] - p1.position[1]
                dist += np.hypot(dx, dy)

            fix_dur = sum(f.duration for f in fixations)
            avg_fix = fix_dur / len(fixations) if fixations else 0
            fix_rate = (len(fixations) / scan_dur) if scan_dur > 0 else 0

            if completion_times and len(completion_times):
                avg_comp = sum(completion_times) / len(completion_times)
                min_comp = min(completion_times)
                max_comp = max(completion_times)
            else:
                avg_comp, min_comp, max_comp = (0, 0, 0)

            # Organize data into labeled sections
            sections = {
                "Temporal Metrics": [
                    f"Total Duration: {scan_dur:.2f}s",
                    f"Avg Completion Time: {avg_comp:.2f}s",
                    f"Min/Max Completion: {min_comp:.2f}s / {max_comp:.2f}s"
                ],
                "Movement Metrics": [
                    f"Avg Velocity: {avg_v:.1f} px/s",
                    f"Peak Velocity: {max_v:.1f} px/s",
                    f"Total Path: {dist:.0f}px"
                ],
                "Fixation Metrics": [
                    f"Number of Fixations: {len(fixations)}",
                    f"Avg Fixation Duration: {avg_fix * 1000:.0f}ms",
                    f"Fixation Rate: {fix_rate:.1f}/s"
                ],
                "Task Metrics": [
                    f"Targets Complete: {len(completion_times or [])} / {len(target_positions)}",
                    f"Task Efficiency: {dist / len(target_positions) if target_positions else 0:.0f}px/target"
                ]
            }

            # Draw each section in the panel
            y_off = r.top + 10
            x_pos = r.left + 10
            for hdr, lines in sections.items():
                hdr_sf = self.parent.FONT.render(hdr, True, self.parent.CHART_COLOR)
                self.parent.screen.blit(hdr_sf, (x_pos, y_off))
                y_off += 25
                for ln in lines:
                    sf = self.parent.FONT.render(ln, True, self.parent.TEXT_COLOR)
                    self.parent.screen.blit(sf, (x_pos + 10, y_off))
                    y_off += 20
                y_off += 10


# --------------------------------------------------------------------------------------
# TimelinePanel Class
# --------------------------------------------------------------------------------------
class TimelinePanel:
    """
    Provides a scrollable timeline view of saccades, fixations, and completion events.
    Allows toggling between saccade and fixation lists, with a scrollbar for navigation.
    """

    def __init__(self, parent):
        """
        Args:
            parent: Reference to the parent (AnalysisView) using this timeline.
        """
        self.parent = parent
        self.open = False                # Whether the timeline panel is visible
        self.mode_saccades = True        # Toggle between displaying saccades vs. fixations
        self.scroll_offset_saccades = 0
        self.scroll_offset_fixations = 0
        self.scroll_offset = 0
        self.dragging = False
        self.drag_offset = 0
        self.panel_rect = parent.expanded_timeline_rect
        self.TIMELINE_ITEM_HEIGHT = 30
        self.track_rect = None
        self.handle_rect = None
        self.handle_range = 0
        self.min_scroll = 0
        self.max_scroll = 0
        self.visible_items = []

    def draw(self, fixations: List[Fixation],
             saccades: List[Saccade],
             completion_times: Optional[List[float]]) -> None:
        """
        Render the timeline panel if open, showing either saccade or fixation items,
        plus any completion time markers.

        Args:
            fixations (List[Fixation]): List of all fixations.
            saccades (List[Saccade]): List of all saccades.
            completion_times (Optional[List[float]]): Times at which targets were completed.
        """
        if not self.open:
            return

        # Keep separate scroll offsets for saccades vs. fixations
        if self.mode_saccades:
            self.scroll_offset = self.scroll_offset_saccades
        else:
            self.scroll_offset = self.scroll_offset_fixations

        r = self.panel_rect
        pygame.draw.rect(self.parent.screen, self.parent.PANEL_COLOR, r)
        pygame.draw.rect(self.parent.screen, self.parent.GRID_COLOR, r, 1)

        # Toggle buttons for saccade vs. fixation mode
        toggle_width = 140
        toggle_height = 30
        rect_saccades = pygame.Rect(r.x + 10, r.y + 10, toggle_width, toggle_height)
        rect_fixations = pygame.Rect(r.x + 10 + toggle_width + 10, r.y + 10, toggle_width, toggle_height)

        # Highlight the active mode button
        if self.mode_saccades:
            pygame.draw.rect(self.parent.screen, (80, 80, 120), rect_saccades)
        else:
            pygame.draw.rect(self.parent.screen, (80, 80, 120), rect_fixations)

        pygame.draw.rect(self.parent.screen, self.parent.BUTTON_BORDER_COLOR, rect_saccades, 2)
        pygame.draw.rect(self.parent.screen, self.parent.BUTTON_BORDER_COLOR, rect_fixations, 2)
        self.parent._draw_clean_text("Saccades", rect_saccades.x + 10, rect_saccades.y + 5)
        self.parent._draw_clean_text("Fixations", rect_fixations.x + 10, rect_fixations.y + 5)

        # Scrollable list area
        list_area_height = r.height - (toggle_height + 20)
        list_x = r.x + 10
        sb_width = 10
        item_width = r.width - sb_width - 20
        list_y = (r.y + 10 + toggle_height + 2) - self.scroll_offset

        content_rect = pygame.Rect(
            list_x,
            r.y + 10 + toggle_height + 2,
            item_width,
            list_area_height
        )
        old_clip = self.parent.screen.get_clip()
        self.parent.screen.set_clip(content_rect)

        # Build items
        self.visible_items.clear()
        all_items = []
        if self.mode_saccades:
            sorted_sac = sorted(saccades, key=lambda s: s.start_time)
            for i, sac in enumerate(sorted_sac):
                all_items.append((sac.start_time, "saccade", f"Saccade #{i + 1}", sac))
            if completion_times:
                for i, ctime in enumerate(completion_times):
                    all_items.append((ctime, "completion", f"Completed at: {ctime}", i))
        else:
            sorted_fix = sorted(fixations, key=lambda f: f.start_time)
            for i, fix in enumerate(sorted_fix):
                all_items.append((fix.start_time, "fixation", f"Fixation #{i + 1}", fix))
            if completion_times:
                for i, ctime in enumerate(completion_times):
                    all_items.append((ctime, "completion", f"Completed at: {ctime}", i))

        all_items.sort(key=lambda x: x[0])

        # Draw the items in the timeline
        for time_val, kind, label, data_obj in all_items:
            item_rect = pygame.Rect(list_x, list_y, item_width, self.TIMELINE_ITEM_HEIGHT)

            # Check if within the visible part of the panel
            if item_rect.bottom >= content_rect.top and item_rect.top <= content_rect.bottom:
                if kind in ("saccade", "fixation"):
                    is_selected = (
                        (kind == "saccade" and data_obj == self.parent.selected_saccade) or
                        (kind == "fixation" and data_obj == self.parent.selected_fixation)
                    )
                    if is_selected:
                        pygame.draw.rect(self.parent.screen, (128, 128, 0), item_rect)  # Olive highlight
                    self.parent._draw_clean_text(label, item_rect.x + 5, item_rect.y + 5)
                    self.visible_items.append((item_rect, kind, data_obj))
                else:  # "completion"
                    is_selected = (data_obj == self.parent.selected_target_index)
                    if is_selected:
                        pygame.draw.rect(self.parent.screen, (128, 128, 0), item_rect)
                    # Draw a simple line for completion
                    line_y = item_rect.centery
                    line_start = (item_rect.x + 10, line_y)
                    line_end = (item_rect.right - 10, line_y)
                    line_color = (200, 180, 60)
                    if is_selected:
                        line_color = (255, 255, 255)  # White for selected
                    pygame.draw.line(self.parent.screen, line_color, line_start, line_end, 2)
                    self.visible_items.append((item_rect, kind, data_obj))

            list_y += self.TIMELINE_ITEM_HEIGHT

        self.parent.screen.set_clip(old_clip)

        # Calculate scroll limits
        total_items = len(all_items)
        content_height = total_items * self.TIMELINE_ITEM_HEIGHT
        self.min_scroll = 0
        self.max_scroll = max(0, content_height - list_area_height)

        # Draw scrollbar if needed
        if content_height > list_area_height:
            sb_x = r.right - sb_width - 5
            sb_y = r.y + 10 + toggle_height + 2
            sb_height = list_area_height
            self.track_rect = pygame.Rect(sb_x, sb_y, sb_width, sb_height)
            pygame.draw.rect(self.parent.screen, self.parent.GRID_COLOR, self.track_rect)

            handle_ratio = list_area_height / float(content_height)
            handle_h = max(int(sb_height * handle_ratio), 20)
            self.handle_range = sb_height - handle_h

            ratio = self.scroll_offset / float(self.max_scroll) if self.max_scroll > 0 else 0
            handle_y = sb_y + int(ratio * self.handle_range)
            self.handle_rect = pygame.Rect(sb_x, handle_y, sb_width, handle_h)
            pygame.draw.rect(self.parent.screen, (100, 100, 140), self.handle_rect)
        else:
            self.track_rect = None
            self.handle_rect = None
            self.handle_range = 0

        # Save the scroll offset
        if self.mode_saccades:
            self.scroll_offset_saccades = self.scroll_offset
        else:
            self.scroll_offset_fixations = self.scroll_offset

    def handle_event(self, event: pygame.event.Event) -> None:
        """
        Process mouse events for toggling modes (saccades vs. fixations), scrolling,
        and item selection (e.g., selecting a fixation or saccade).

        Args:
            event (pygame.event.Event): The event to handle.
        """
        mouse_pos = pygame.mouse.get_pos()
        r = self.panel_rect
        toggle_width = 140
        toggle_height = 30
        rect_sac = pygame.Rect(r.x + 10, r.y + 10, toggle_width, toggle_height)
        rect_fix = pygame.Rect(r.x + 10 + toggle_width + 10, r.y + 10, toggle_width, toggle_height)

        # Mode toggle
        if event.type == pygame.MOUSEBUTTONDOWN:
            if rect_sac.collidepoint(mouse_pos):
                self.mode_saccades = True
                self.parent.selected_fixation = None
                self.parent.selected_target_index = None
                return
            elif rect_fix.collidepoint(mouse_pos):
                self.mode_saccades = False
                self.parent.selected_saccade = None
                self.parent.selected_target_index = None
                return

            if r.collidepoint(mouse_pos):
                # Mouse wheel scrolling
                if event.button in (4, 5):
                    if self.mode_saccades:
                        self.scroll_offset = self.scroll_offset_saccades
                    else:
                        self.scroll_offset = self.scroll_offset_fixations

                    speed = 20
                    if event.button == 4:  # Wheel up
                        self.scroll_offset -= speed
                    else:  # Wheel down
                        self.scroll_offset += speed

                    self._clamp_scroll()

                    if self.mode_saccades:
                        self.scroll_offset_saccades = self.scroll_offset
                    else:
                        self.scroll_offset_fixations = self.scroll_offset

                    # Deselect target when scrolling
                    self.parent.selected_target_index = None
                    return

                # Check for scrollbar dragging
                if self.handle_rect and self.handle_rect.collidepoint(mouse_pos):
                    self.dragging = True
                    self.drag_offset = mouse_pos[1] - self.handle_rect.y
                    return

                if self.track_rect and self.track_rect.collidepoint(mouse_pos):
                    new_y = mouse_pos[1] - self.track_rect.y
                    if self.handle_rect:
                        new_y -= self.handle_rect.height // 2
                        new_y = max(0, min(self.handle_range, new_y))
                        self.handle_rect.y = self.track_rect.y + new_y
                        self._update_scroll_from_handle()
                    return

                # Check if user clicked on a timeline item
                for item_rect, kind, data_obj in self.visible_items:
                    if item_rect.collidepoint(mouse_pos):
                        if kind == "saccade":
                            self.parent.selected_saccade = data_obj
                            self.parent.selected_fixation = None
                            self.parent.selected_target_index = None
                        elif kind == "fixation":
                            self.parent.selected_fixation = data_obj
                            self.parent.selected_saccade = None
                            self.parent.selected_target_index = None
                        elif kind == "completion":
                            # data_obj is the completion index
                            if self.parent.selected_target_index == data_obj:
                                self.parent.selected_target_index = None
                            else:
                                self.parent.selected_target_index = data_obj
                            self.parent.selected_saccade = None
                            self.parent.selected_fixation = None
                        return

        elif event.type == pygame.MOUSEWHEEL:
            # Another way to scroll using the standard MOUSEWHEEL event
            if r.collidepoint(mouse_pos):
                if self.mode_saccades:
                    self.scroll_offset = self.scroll_offset_saccades
                else:
                    self.scroll_offset = self.scroll_offset_fixations

                speed = 20
                self.scroll_offset -= event.y * speed

                self._clamp_scroll()

                if self.mode_saccades:
                    self.scroll_offset_saccades = self.scroll_offset
                else:
                    self.scroll_offset_fixations = self.scroll_offset

                # Clear target selection when scrolling
                self.parent.selected_target_index = None
                return

        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
            self.drag_offset = 0

        elif event.type == pygame.MOUSEMOTION:
            if self.dragging and self.handle_rect and self.track_rect:
                new_y = mouse_pos[1] - self.drag_offset
                new_y = max(self.track_rect.y, min(self.track_rect.bottom - self.handle_rect.height, new_y))
                self.handle_rect.y = new_y
                self._update_scroll_from_handle()

    # ------------------------------------------------------------------------
    # Internal Helper Methods
    # ------------------------------------------------------------------------
    def _update_scroll_from_handle(self) -> None:
        """
        Update the timeline's scroll offset based on the handle's position in the track.
        """
        if not self.handle_rect or not self.track_rect or self.handle_range <= 0:
            return
        handle_pos = self.handle_rect.y - self.track_rect.y
        ratio = handle_pos / float(self.handle_range)
        self.scroll_offset = ratio * self.max_scroll
        self._clamp_scroll()

        if self.mode_saccades:
            self.scroll_offset_saccades = self.scroll_offset
        else:
            self.scroll_offset_fixations = self.scroll_offset

    def _clamp_scroll(self) -> None:
        """
        Prevent scrolling beyond the minimum or maximum content extents.
        """
        if self.scroll_offset < self.min_scroll:
            self.scroll_offset = self.min_scroll
        elif self.scroll_offset > self.max_scroll:
            self.scroll_offset = self.max_scroll


class FilterPanel:
    """
    A draggable panel for filtering fixations and saccades based on duration or distance.
    Provides slider-based thresholds to limit data displayed in the analysis view.
    """

    def __init__(self, parent):
        """
        Args:
            parent: Reference to the main analysis view or parent component.
        """
        self.parent = parent
        self.open = False

        # Panel rectangle anchored to the right side
        self.panel_rect = pygame.Rect(
            self.parent.width - 220,
            170,
            200,
            220
        )

        # Sliders for fixation duration and saccade distance
        self.fixation_slider_val = 0.0
        self.saccade_slider_val = 0.0

        # Data-driven min/max
        self.fixation_min = 0.0
        self.fixation_max = 1.0
        self.saccade_min = 0.0
        self.saccade_max = 1.0

        # Layout constants
        self.TITLE_PADDING = 15
        self.LABEL_SLIDER_GAP = 40
        self.SLIDER_VALUE_GAP = 12
        self.SECTION_GAP = 55

        # Vertical layout for fixation and saccade sections
        self.fix_label_y = self.panel_rect.y + 15
        self.fix_slider_y = self.fix_label_y + self.LABEL_SLIDER_GAP
        self.sac_label_y = self.fix_slider_y + 45
        self.sac_slider_y = self.sac_label_y + self.LABEL_SLIDER_GAP

        # Slider rectangles
        self.fix_slider_rect = pygame.Rect(
            self.panel_rect.x + 20,
            self.fix_slider_y,
            160,
            8
        )
        self.sac_slider_rect = pygame.Rect(
            self.panel_rect.x + 20,
            self.sac_slider_y,
            160,
            8
        )

        # Drag state
        self.dragging_fixation = False
        self.dragging_saccade = False

    def update_data_ranges(self, fixations: List[Fixation], saccades: List[Saccade]) -> None:
        """
        Refresh slider ranges based on actual min/max values from the data.

        Args:
            fixations (List[Fixation]): Current fixations to find min/max durations.
            saccades (List[Saccade]): Current saccades to find min/max distances.
        """
        if fixations:
            durations = [f.duration for f in fixations]
            self.fixation_min = min(durations)
            self.fixation_max = max(durations)
        else:
            self.fixation_min = 0.0
            self.fixation_max = 1.0

        if saccades:
            distances = [s.distance_traveled for s in saccades]
            self.saccade_min = min(distances)
            self.saccade_max = max(distances)
        else:
            self.saccade_min = 0.0
            self.saccade_max = 1.0

        # Avoid zero-range issues
        if abs(self.fixation_max - self.fixation_min) < 1e-9:
            self.fixation_max += 1.0
        if abs(self.saccade_max - self.saccade_min) < 1e-9:
            self.saccade_max += 1.0

    def draw(self, fixations: List[Fixation], saccades: List[Saccade]) -> None:
        """
        Draw the filter panel with its sliders if currently open.

        Args:
            fixations (List[Fixation]): Not used here directly, but included for context.
            saccades (List[Saccade]): Same as fixations.
        """
        if not self.open:
            return

        # Panel background & border
        pygame.draw.rect(self.parent.screen, self.parent.PANEL_COLOR, self.panel_rect)
        pygame.draw.rect(self.parent.screen, self.parent.GRID_COLOR, self.panel_rect, 1)

        # Labels
        fix_label = self.parent.FONT.render("Fixation Duration", True, self.parent.TEXT_COLOR)
        sac_label = self.parent.FONT.render("Saccade Distance", True, self.parent.TEXT_COLOR)
        self.parent.screen.blit(fix_label, (self.panel_rect.x + 20, self.fix_label_y))
        self.parent.screen.blit(sac_label, (self.panel_rect.x + 20, self.sac_label_y))

        # Slider tracks
        line_color = (150, 150, 150)
        track_thickness = 4
        # Fixation
        pygame.draw.line(
            self.parent.screen,
            line_color,
            (self.fix_slider_rect.left, self.fix_slider_rect.centery),
            (self.fix_slider_rect.right, self.fix_slider_rect.centery),
            track_thickness
        )
        # Saccade
        pygame.draw.line(
            self.parent.screen,
            line_color,
            (self.sac_slider_rect.left, self.sac_slider_rect.centery),
            (self.sac_slider_rect.right, self.sac_slider_rect.centery),
            track_thickness
        )

        # Slider handles
        handle_color = (200, 200, 50)
        handle_width = 10
        handle_height = 20

        # Fixation handle position
        handle_x_fix = int(self.fix_slider_rect.left + self.fixation_slider_val * self.fix_slider_rect.width)
        handle_rect_fix = pygame.Rect(
            handle_x_fix - handle_width // 2,
            self.fix_slider_rect.centery - handle_height // 2,
            handle_width,
            handle_height
        )
        pygame.draw.rect(self.parent.screen, handle_color, handle_rect_fix)
        pygame.draw.rect(self.parent.screen, (255, 255, 255), handle_rect_fix, 1)

        # Saccade handle position
        handle_x_sac = int(self.sac_slider_rect.left + self.saccade_slider_val * self.sac_slider_rect.width)
        handle_rect_sac = pygame.Rect(
            handle_x_sac - handle_width // 2,
            self.sac_slider_rect.centery - handle_height // 2,
            handle_width,
            handle_height
        )
        pygame.draw.rect(self.parent.screen, handle_color, handle_rect_sac)
        pygame.draw.rect(self.parent.screen, (255, 255, 255), handle_rect_sac, 1)

        # Current slider values
        fix_val = self.fixation_min + self.fixation_slider_val * (self.fixation_max - self.fixation_min)
        fix_val_str = f"{fix_val:.2f}s"
        sac_val = self.saccade_min + self.saccade_slider_val * (self.saccade_max - self.saccade_min)
        sac_val_str = f"{sac_val:.2f}px"

        fix_val_surf = self.parent.FONT.render(fix_val_str, True, self.parent.TEXT_COLOR)
        sac_val_surf = self.parent.FONT.render(sac_val_str, True, self.parent.TEXT_COLOR)

        # Render the numerical values below each slider
        self.parent.screen.blit(fix_val_surf,
                                (self.fix_slider_rect.x,
                                 self.fix_slider_rect.bottom + self.SLIDER_VALUE_GAP))
        self.parent.screen.blit(sac_val_surf,
                                (self.sac_slider_rect.x,
                                 self.sac_slider_rect.bottom + self.SLIDER_VALUE_GAP))

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Manage mouse events for slider interaction and dragging.

        Args:
            event (pygame.event.Event): The event to process.

        Returns:
            bool: True if the panel consumed this event, False otherwise.
        """
        if not self.open:
            return False

        mouse_pos = pygame.mouse.get_pos()
        handle_width = 10
        handle_height = 20

        # Rebuild handle rects for both sliders
        handle_x_fix = int(self.fix_slider_rect.left + self.fixation_slider_val * self.fix_slider_rect.width)
        handle_rect_fix = pygame.Rect(
            handle_x_fix - handle_width // 2,
            self.fix_slider_rect.centery - handle_height // 2,
            handle_width,
            handle_height
        )

        handle_x_sac = int(self.sac_slider_rect.left + self.saccade_slider_val * self.sac_slider_rect.width)
        handle_rect_sac = pygame.Rect(
            handle_x_sac - handle_width // 2,
            self.sac_slider_rect.centery - handle_height // 2,
            handle_width,
            handle_height
        )

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Check for direct click on fixation slider track
            if self.fix_slider_rect.collidepoint(mouse_pos):
                ratio = (mouse_pos[0] - self.fix_slider_rect.left) / float(self.fix_slider_rect.width)
                self.fixation_slider_val = max(0.0, min(1.0, ratio))
                self.dragging_fixation = True
                return True
            # Check for direct click on saccade slider track
            elif self.sac_slider_rect.collidepoint(mouse_pos):
                ratio = (mouse_pos[0] - self.sac_slider_rect.left) / float(self.sac_slider_rect.width)
                self.saccade_slider_val = max(0.0, min(1.0, ratio))
                self.dragging_saccade = True
                return True
            # Check handle clicks
            elif handle_rect_fix.collidepoint(mouse_pos):
                self.dragging_fixation = True
                return True
            elif handle_rect_sac.collidepoint(mouse_pos):
                self.dragging_saccade = True
                return True

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            # Stop dragging on mouse release
            was_dragging = self.dragging_fixation or self.dragging_saccade
            self.dragging_fixation = False
            self.dragging_saccade = False
            return was_dragging

        elif event.type == pygame.MOUSEMOTION:
            # Drag fixation slider
            if self.dragging_fixation:
                ratio = (mouse_pos[0] - self.fix_slider_rect.left) / float(self.fix_slider_rect.width)
                self.fixation_slider_val = max(0.0, min(1.0, ratio))
                return True
            # Drag saccade slider
            elif self.dragging_saccade:
                ratio = (mouse_pos[0] - self.sac_slider_rect.left) / float(self.sac_slider_rect.width)
                self.saccade_slider_val = max(0.0, min(1.0, ratio))
                return True

        return False

    def get_fixation_cutoff(self) -> float:
        """
        Compute the current fixation duration threshold from the slider.

        Returns:
            float: Minimum fixation duration in seconds.
        """
        return self.fixation_min + self.fixation_slider_val * (self.fixation_max - self.fixation_min)

    def get_saccade_cutoff(self) -> float:
        """
        Compute the current saccade distance threshold from the slider.

        Returns:
            float: Minimum saccade travel distance in pixels.
        """
        return self.saccade_min + self.saccade_slider_val * (self.saccade_max - self.saccade_min)





