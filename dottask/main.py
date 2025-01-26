#main.py

# =========================================================================================
# MAIN.PY
#
# Entry point for the gaze tracking application for conducting the dot task. Manages the main loop, handling transitions
# between verification and analysis views, collecting gaze data, and performing offline
# analysis (fixations, saccades, heatmaps). Integrates with an eye tracker via the
# Tobii Research SDK.
# =========================================================================================



import pygame
import time
import tobii_research as tr
from views import VerificationView, AnalysisView, HeatmapView
from processors import GazeProcessor, TargetProcessor, GazeAnalyzer

class GazeAuth:
    """
    The main application class for the gaze tracker. It initializes Pygame,
    sets up the display, manages different views (verification, analysis, heatmap),
    and orchestrates the workflow between data capture and analysis.
    """

    def __init__(self):
        """
        Set up Pygame, initialize the display, and create all required views and processors.
        Establishes the default state for verification mode and loads target positions.
        """
        # -------------------------------
        # 1) Pygame Initialization
        # -------------------------------
        pygame.init()
        pygame.display.set_caption("Gaze Tracker")
        
        # Detect current display info
        display_info = pygame.display.Info()
        self.width = display_info.current_w
        self.height = display_info.current_h

        # Create a fullscreen window
        self.screen = pygame.display.set_mode(
            (self.width, self.height),
            pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF,
            32
        )
        
        # -------------------------------
        # 2) Views Initialization
        # -------------------------------
        self.verification_view = VerificationView(self.screen, self.width, self.height)
        self.analysis_view = AnalysisView(self.screen, self.width, self.height)
        self.heatmap_view = HeatmapView(self.screen, self.width, self.height)

        # Current state: verification phase
        self.current_view = 'verification'
        # Switch between 'analysis' and 'heatmap' within the analysis view
        self.analysis_mode = 'analysis'
        
        # -------------------------------
        # 3) Processors Initialization
        # -------------------------------
        self.gaze_processor = GazeProcessor(self.width, self.height)
        self.target_processor = TargetProcessor(self.width, self.height)
        
        # 4) Eye tracker discovery
        self.eyetracker = self.initialize_eyetracker()
        
        # 5) Initialize target positions
        self.target_processor.generate_positions(5)  # Creates five target dots
        
        # -------------------------------
        # 6) Variables for Analysis
        # -------------------------------
        self.gaze_analyzer = None
        self.fixations = []
        self.saccades = []
        self.metrics = {}
        self.heatmap_data = {}
        self.completion_relative_times = []
        self.completion_times = []

    def initialize_eyetracker(self) -> tr.EyeTracker:
        """
        Find and return the first available Tobii eye tracker device.

        Raises:
            RuntimeError: If no eye trackers are found.

        Returns:
            tr.EyeTracker: The discovered eye tracker device.
        """
        found_eyetrackers = tr.find_all_eyetrackers()
        if not found_eyetrackers:
            raise RuntimeError("No eye trackers found")
        return found_eyetrackers[0]

    def gaze_data_callback(self, gaze_data: dict) -> None:
        """
        Callback for Tobii's gaze data subscription. Feeds raw data into GazeProcessor
        if we are in the verification phase.

        Args:
            gaze_data (dict): Gaze data from the eyetracker, containing left/right eye info.
        """
        if self.current_view == 'verification':
            self.gaze_processor.process_gaze_data(gaze_data)

    def perform_analysis(self) -> None:
        """
        Once verification is done and we have raw gaze data, build a GazeAnalyzer
        to detect fixations/saccades, compute metrics, and start generating a heatmap.
        """
        self.gaze_analyzer = GazeAnalyzer(
            gaze_points=self.gaze_processor.raw_gaze_points,
            fixation_threshold=30,
            fixation_duration_threshold=0.1,
            saccade_velocity_threshold=300
        )
        
        # Collect and store the results
        self.fixations = self.gaze_analyzer.fixations
        self.saccades = self.gaze_analyzer.saccades
        self.metrics = self.gaze_analyzer.calculate_metrics()
        self.heatmap_data = self.gaze_analyzer.heatmap_data
        
        # Use target processor’s times for later reference
        self.completion_times_absolute = self.target_processor.completion_times_absolute
        self.completion_times = self.target_processor.completion_times
        
        # Immediately start building a heatmap in the background
        self.heatmap_view.start_generation(self.gaze_processor.raw_gaze_points)

    def reset_session(self) -> None:
        """
        Reset the entire app state for a fresh session: clears data, re-initializes
        the processors/views, and sets everything back to verification mode.
        """
        # Clear out heatmap data
        self.heatmap_view.clear()
        
        # Reset the view mode
        self.current_view = 'verification'
        self.analysis_mode = 'analysis'
        
        # Regenerate targets and reset trackers
        self.target_processor.generate_positions(5)
        self.gaze_processor.reset()
        self.target_processor.reset()
        
        # Clear analysis results
        self.gaze_analyzer = None
        self.fixations = []
        self.saccades = []
        self.metrics = {}
        self.heatmap_data = {}
        self.completion_relative_times = []
        self.completion_times = []

    def handle_verification_view(self) -> None:
        """
        Primary logic for the verification phase. Checks if user has fixated on the current target
        long enough. If all done, stops recording and moves to analysis.
        """
        if self.target_processor.check_gaze(self.gaze_processor.current_gaze):
            # The user completed the current target
            self.gaze_processor.stop_recording()
            self.perform_analysis()
            self.current_view = 'analysis'
        
        # Animate target changes
        self.target_processor.update_animation()
        
        # Identify the current target dot, if there is one
        current_dot = (
            self.target_processor.positions[self.target_processor.current_idx]
            if self.target_processor.current_idx < len(self.target_processor.positions)
            else None
        )
        
        # Draw verification view
        self.verification_view.draw(
            current_gaze=self.gaze_processor.current_gaze,
            dot_position=current_dot,
            dot_alpha=self.target_processor.alpha,
            dot_size_multiplier=self.target_processor.size_multiplier,
            gaze_history=self.gaze_processor.gaze_history,
            remaining_points=len(self.target_processor.positions) - self.target_processor.current_idx
        )

    def handle_analysis_view(self) -> None:
        """
        Logic for the analysis/heatmap view, switching between analytics visuals
        or the heatmap based on `analysis_mode`.
        """
        if self.analysis_mode == 'analysis':
            # Update the analysis view (handles animations, etc.)
            self.analysis_view.update(1/60)
            self.analysis_view.draw(
                gaze_points=self.gaze_processor.raw_gaze_points,
                fixations=self.fixations,
                saccades=self.saccades,
                heatmap_data=self.heatmap_data,
                target_positions=self.target_processor.positions,
                completion_times=self.completion_times,
                completion_times_absolute=self.target_processor.completion_times_absolute
            )
        else:
            # In heatmap mode, draw the heatmap
            self.heatmap_view.draw(
                self.gaze_processor.raw_gaze_points,
                self.target_processor.positions,
                self.completion_times
            )

    def handle_mode_switch(self) -> None:
        """
        Toggle between analysis mode (graphs, fixations, saccades) and the heatmap view
        without resetting or losing progress in the heatmap generator.
        """
        self.analysis_mode = 'heatmap' if self.analysis_mode == 'analysis' else 'analysis'

    def cleanup(self) -> None:
        """
        Cleanup function called before exit. Stops heatmap generation, unsubscribes from
        Tobii gaze data, and quits Pygame.
        """
        self.heatmap_view.clear()
        self.eyetracker.unsubscribe_from(
            tr.EYETRACKER_GAZE_DATA,
            self.gaze_data_callback
        )
        pygame.quit()

    def run(self) -> None:
        """
        The main application loop. Subscribes to the eye tracker and continuously
        handles events from the OS/Pygame, delegating to the current view.
        """
        # Subscribe to gaze data for continuous updates
        self.eyetracker.subscribe_to(
            tr.EYETRACKER_GAZE_DATA,
            self.gaze_data_callback,
            as_dictionary=True
        )
        
        try:
            running = True
            while running:
                # Process Pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        continue

                    if event.type == pygame.KEYDOWN:
                        # Escape => exit
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        # F => toggle between analysis/heatmap
                        elif event.key == pygame.K_f and self.current_view != 'verification':
                            self.handle_mode_switch()
                        # SPACE => reset session if not in verification
                        elif event.key == pygame.K_SPACE and self.current_view != 'verification':
                            self.reset_session()
                    
                    # Pass mouse events to the analysis view if we're in analysis mode
                    elif event.type == pygame.MOUSEBUTTONDOWN and self.current_view != 'verification':
                        if self.analysis_mode == 'analysis':
                            self.analysis_view.handle_event(event)
                
                # Update the current view
                if self.current_view == 'verification':
                    self.handle_verification_view()
                else:
                    self.handle_analysis_view()
                
                # Framerate limit to ~60 FPS
                time.sleep(1 / 60)
                
        finally:
            self.cleanup()


if __name__ == "__main__":
    visualizer = GazeAuth()
    visualizer.run()