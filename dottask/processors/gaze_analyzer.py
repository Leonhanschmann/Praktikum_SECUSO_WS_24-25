# processors/gaze_analyzer.py


# =========================================================================================
# GAZE_ANALYZER.PY
# 
# Provides classes and methods for analyzing recorded gaze data to identify fixations, saccades, and other metrics.
# 
# =========================================================================================


from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np

@dataclass
class GazePoint:
    """
    Represents a single gaze point measurement.

    Attributes:
        timestamp (float): The time when this gaze point was recorded.
        position (Tuple[int, int]): (x, y) screen coordinates of the gaze.
        velocity (float): Estimated velocity at this point in pixels/second.
    """
    timestamp: float
    position: Tuple[int, int]
    velocity: float

@dataclass
class Fixation:
    """
    Represents a period of stable gaze.

    Attributes:
        start_time (float): Relative or analysis-based start time of the fixation.
        end_time (float): Relative or analysis-based end time of the fixation.
        start_time_absolut (float): Actual (absolute) start timestamp, same as raw data.
        end_time_absolut (float): Actual (absolute) end timestamp, same as raw data.
        center_position (Tuple[int,int]): The centroid (x, y) of the fixation.
        duration (float): Duration of the fixation in seconds.
        gaze_points (List[GazePoint]): List of raw gaze points contained in the fixation.
    """
    start_time: float
    end_time: float
    start_time_absolut: float
    end_time_absolut: float
    center_position: Tuple[int, int]
    duration: float
    gaze_points: List['GazePoint']

@dataclass
class SaccadeSegment:
    """
    Represents one sub-segment of a saccade (phase) grouped by velocity bin (low/medium/high).

    Attributes:
        start_position (Tuple[int,int]): Screen coordinates where this segment begins.
        end_position (Tuple[int,int]): Screen coordinates where this segment ends.
        mean_velocity (float): Average velocity within this segment.
        peak_velocity (float): Maximum velocity within this segment.
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
    Represents rapid eye movement between fixations.

    Attributes:
        start_time (float): Relative or analysis-based start time of the saccade.
        end_time (float): Relative or analysis-based end time of the saccade.
        start_time_absolut (float): Actual (absolute) start timestamp.
        end_time_absolut (float): Actual (absolute) end timestamp.
        start_position (Tuple[int,int]): (x, y) of where the saccade begins.
        end_position (Tuple[int,int]): (x, y) of where the saccade ends.
        duration (float): Duration of the saccade in seconds.
        peak_velocity (float): Maximum velocity of the saccade in pixels/second.
        mean_velocity (float): Mean velocity of the saccade in pixels/second.
        amplitude (float): Euclidean distance between start and end positions in pixels.
        distance_traveled (float): Total path length (sum of point-to-point distances).
        gaze_points (List[GazePoint]): Sequence of gaze points forming the saccade.
        segments (List[SaccadeSegment]): Optional velocity-based sub-segments of the saccade.
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
    segments: List[SaccadeSegment]


class GazeAnalyzer:
    """
    Analyzes recorded gaze data to identify fixations, saccades, and other metrics.
    Works with raw gaze data after recording is complete.
    """

    def __init__(self, 
                 gaze_points: List[GazePoint],
                 fixation_threshold: float = 30,
                 fixation_duration_threshold: float = 0.1,
                 saccade_velocity_threshold: float = 300):
        """
        Args:
            gaze_points (List[GazePoint]): List of raw gaze points to analyze.
            fixation_threshold (float): Distance threshold for fixation detection (pixels).
            fixation_duration_threshold (float): Minimum duration for a valid fixation (seconds).
            saccade_velocity_threshold (float): Velocity threshold for saccade detection (pixels/second).
        """
        # Analysis parameters
        self.FIXATION_THRESHOLD = fixation_threshold
        self.FIXATION_DURATION_THRESHOLD = fixation_duration_threshold
        self.SACCADE_VELOCITY_THRESHOLD = saccade_velocity_threshold
        
        # Analysis results
        self.fixations: List[Fixation] = []
        self.saccades: List[Saccade] = []
        self.heatmap_data: Dict[Tuple[int, int], int] = {}
        
        # Start the gaze analysis immediately
        self._analyze_gaze_data(gaze_points)
        
    def _analyze_gaze_data(self, gaze_points: List[GazePoint]) -> None:
        """
        Internal method to perform the complete analysis of the provided gaze data,
        including event (fixation/saccade) detection and heatmap generation.
        """
        self._detect_events(gaze_points)
        self._generate_heatmap(gaze_points)
        
    def _detect_events(self, gaze_points: List[GazePoint]) -> None:
        """
        Detect fixations and saccades in the gaze data using basic thresholding heuristics.
        
        This function splits the data into consecutive segments classified as either 
        fixation or saccade based on velocity thresholds and position dispersion.
        """
        self.fixations = []
        self.saccades = []
        
        if not gaze_points:
            return
        
        current_segment: List[GazePoint] = []
        in_fixation = False
        
        for point in gaze_points:
            # If velocity is above threshold => likely saccade
            if point.velocity > self.SACCADE_VELOCITY_THRESHOLD:
                if in_fixation:
                    # End current fixation if it meets minimum length
                    if len(current_segment) >= 3:
                        self._create_fixation(current_segment)
                    current_segment = []
                    in_fixation = False
                
                # Add the point to the current saccade segment
                current_segment.append(point)
            else:
                # If velocity is below threshold => likely fixation
                if not in_fixation:
                    # End the current saccade
                    if len(current_segment) >= 2:
                        self._create_saccade(current_segment)
                    current_segment = []
                    in_fixation = True
                
                # Add to the fixation
                current_segment.append(point)
                
                # Check for position dispersion to confirm or break fixation
                if len(current_segment) >= 3:
                    center = self._calculate_centroid([p.position for p in current_segment])
                    distance = self._calculate_distance(point.position, center)
                    
                    # If distance from centroid exceeds fixation threshold => break
                    if distance > self.FIXATION_THRESHOLD:
                        # End current fixation
                        if len(current_segment) >= 3:
                            self._create_fixation(current_segment[:-1])
                        current_segment = [point]
        
        # Handle the last segment after iteration
        if len(current_segment) >= 3 and in_fixation:
            self._create_fixation(current_segment)
        elif len(current_segment) >= 2 and not in_fixation:
            self._create_saccade(current_segment)
    
    def _create_fixation(self, points: List[GazePoint]) -> None:
        """
        Create a fixation from a sequence of gaze points if it meets 
        the minimum duration threshold. Also updates the heatmap intensity.

        Args:
            points (List[GazePoint]): A series of consecutive low-velocity points.
        """
        duration = points[-1].timestamp - points[0].timestamp
        
        if duration >= self.FIXATION_DURATION_THRESHOLD:
            center = self._calculate_centroid([p.position for p in points])
            
            fixation = Fixation(
                start_time=points[0].timestamp,
                end_time=points[-1].timestamp,
                start_time_absolut=points[0].timestamp,
                end_time_absolut=points[-1].timestamp,
                center_position=center,
                duration=duration,
                gaze_points=points
            )
            self.fixations.append(fixation)
            
            # Increase heatmap intensity around the fixation center
            pos = (center[0] // 20 * 20, center[1] // 20 * 20)
            self.heatmap_data[pos] = self.heatmap_data.get(pos, 0) + 5
    
    def _create_saccade(self, points: List[GazePoint]) -> None:
        """
        Create a saccade from a series of gaze points. Also splits the saccade
        into velocity-based segments for deeper analysis.

        Args:
            points (List[GazePoint]): A series of consecutive high-velocity points.
        """
        if len(points) < 2:
            return
        
        # 1) Calculate top-level saccade metrics
        start_time = points[0].timestamp
        end_time = points[-1].timestamp
        start_pos = points[0].position
        end_pos = points[-1].position
        duration = end_time - start_time
        peak_velocity = max(p.velocity for p in points)
        mean_velocity = sum(p.velocity for p in points) / len(points) if points else 0
        amplitude = self._calculate_distance(start_pos, end_pos)
        distance_traveled = self._calculate_path_length([p.position for p in points])

        # 2) Create velocity-based sub-segments (low/medium/high bins)
        velocities = [p.velocity for p in points]
        min_vel = min(velocities)
        max_vel = max(velocities)

        if min_vel == max_vel:
            # All velocities are the same => one segment
            segment_peak_velocity = max(velocities)
            segment_gaze_point_count = len(points)
            segments = [
                SaccadeSegment(
                    start_position=start_pos,
                    end_position=end_pos,
                    mean_velocity=min_vel,
                    peak_velocity=segment_peak_velocity,
                    gaze_point_count=segment_gaze_point_count
                )
            ]
        else:
            # Divide velocity range into three equal bins
            bin1 = min_vel + (max_vel - min_vel) / 3.0
            bin2 = min_vel + 2.0 * (max_vel - min_vel) / 3.0

            segments: List[SaccadeSegment] = []
            seg_start_index = 0
            current_bin = None

            def which_bin(v: float) -> int:
                """Return bin index: 0=LOW, 1=MEDIUM, 2=HIGH."""
                if v < bin1:
                    return 0
                elif v < bin2:
                    return 1
                else:
                    return 2

            # Go through pairs of GazePoints to group segments
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]
                mean_vel_segment = 0.5 * (p1.velocity + p2.velocity)
                b = which_bin(mean_vel_segment)

                if current_bin is None:
                    # Start first segment
                    current_bin = b
                    seg_start_index = i
                else:
                    # Check if bin changed => close old segment and start new
                    if b != current_bin:
                        seg_points = points[seg_start_index : i + 1]
                        seg_start = seg_points[0].position
                        seg_end = seg_points[-1].position
                        seg_mean_vel = sum(p.velocity for p in seg_points) / len(seg_points) if seg_points else min_vel
                        seg_peak_vel = max(p.velocity for p in seg_points) if seg_points else min_vel
                        seg_gaze_point_count = len(seg_points)

                        segments.append(
                            SaccadeSegment(
                                start_position=seg_start,
                                end_position=seg_end,
                                mean_velocity=seg_mean_vel,
                                peak_velocity=seg_peak_vel,
                                gaze_point_count=seg_gaze_point_count
                            )
                        )
                        current_bin = b
                        seg_start_index = i

            # Close the last segment
            seg_points = points[seg_start_index:]
            seg_start = seg_points[0].position
            seg_end = seg_points[-1].position
            seg_mean_vel = sum(p.velocity for p in seg_points) / len(seg_points) if seg_points else min_vel
            seg_peak_vel = max(p.velocity for p in seg_points) if seg_points else min_vel
            seg_gaze_point_count = len(seg_points)

            segments.append(
                SaccadeSegment(
                    start_position=seg_start,
                    end_position=seg_end,
                    mean_velocity=seg_mean_vel,
                    peak_velocity=seg_peak_vel,
                    gaze_point_count=seg_gaze_point_count
                )
            )

        # 3) Create the final Saccade object
        saccade = Saccade(
            start_time=start_time,
            end_time=end_time,
            start_time_absolut=start_time,
            end_time_absolut=end_time,
            start_position=start_pos,
            end_position=end_pos,
            duration=duration,
            peak_velocity=peak_velocity,
            mean_velocity=mean_velocity,
            amplitude=amplitude,
            distance_traveled=distance_traveled,
            gaze_points=points,
            segments=segments
        )
        self.saccades.append(saccade)
    
    def _generate_heatmap(self, gaze_points: List[GazePoint]) -> None:
        """
        Generate a simple heatmap representation by counting the occurrences of 
        gaze positions in (20x20) pixel buckets.

        Args:
            gaze_points (List[GazePoint]): All raw gaze points.
        """
        self.heatmap_data.clear()
        for point in gaze_points:
            pos = (point.position[0] // 20 * 20, point.position[1] // 20 * 20)
            self.heatmap_data[pos] = self.heatmap_data.get(pos, 0) + 1
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate and return various gaze metrics from fixations and saccades.

        Returns:
            Dict: A dictionary containing aggregated metrics like:
                  'number_of_fixations', 'mean_fixation_duration', etc.
                  Returns an empty dict if no fixations or saccades exist.
        """
        if not self.fixations or not self.saccades:
            return {}
            
        metrics = {
            'number_of_fixations': len(self.fixations),
            'number_of_saccades': len(self.saccades),
            'mean_fixation_duration': np.mean([f.duration for f in self.fixations]),
            'total_fixation_time': sum(f.duration for f in self.fixations),
            'mean_saccade_amplitude': np.mean([s.amplitude for s in self.saccades]),
            'mean_saccade_velocity': np.mean([s.peak_velocity for s in self.saccades]),
            'scan_path_length': sum(s.amplitude for s in self.saccades),
            'mean_saccade_mean_velocity': np.mean([s.mean_velocity for s in self.saccades]),
            'total_saccade_distance': sum(s.distance_traveled for s in self.saccades)
        }
        
        # Additional temporal metrics
        if self.fixations and self.saccades:
            total_time = self.fixations[-1].end_time - self.fixations[0].start_time
            metrics.update({
                'total_scan_time': total_time,
                'fixation_frequency': len(self.fixations) / total_time if total_time > 0 else 0
            })
            
        return metrics
    
    # ------------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------------
    
    def _calculate_centroid(self, positions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Calculate the centroid of a list of (x, y) positions.

        Args:
            positions (List[Tuple[int,int]]): A list of gaze positions.

        Returns:
            Tuple[int,int]: The integer coordinates of the centroid.
        """
        x_sum = sum(x for x, _ in positions)
        y_sum = sum(y for _, y in positions)
        count = len(positions)
        return (int(x_sum / count), int(y_sum / count))
    
    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Calculate the Euclidean distance between two points (pos1, pos2).

        Args:
            pos1 (Tuple[int,int]): Starting position.
            pos2 (Tuple[int,int]): Ending position.

        Returns:
            float: Euclidean distance in pixels.
        """
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        return np.sqrt(dx * dx + dy * dy)
    
    def _calculate_path_length(self, positions: List[Tuple[int, int]]) -> float:
        """
        Calculate the total path length through a series of positions.

        Args:
            positions (List[Tuple[int,int]]): List of positions representing a path.

        Returns:
            float: Sum of consecutive distances in pixels.
        """
        if len(positions) < 2:
            return 0.0
        
        return sum(
            self._calculate_distance(pos1, pos2)
            for pos1, pos2 in zip(positions[:-1], positions[1:])
        )
