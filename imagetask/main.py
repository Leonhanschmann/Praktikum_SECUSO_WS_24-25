# =========================================================================================
# MAIN.PY
#
# Entry point for the GazeImageViewer application. 
# Initializes the application, manages the main loop, handles events, 
# and coordinates between different views and the gaze processor.
# Integrates with a Tobii eye-tracking device  to capture real-time 
# gaze data and visualize it over a sequence of images as heat maps. 
# =========================================================================================



import os
import time
import logging
from pathlib import Path

import numpy as np  
import pygame
import tobii_research as tr  # Requires Tobii SDK.

# Configure logging to capture and display debug information
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture detailed logs
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Import custom views and processors
from views.image_task_view import ImageTaskView
from views.image_analysis_view import ImageAnalysisView
from processors.gaze_processor import GazeProcessor


def compute_rms_brightness(image_path: str) -> float:
    """
    Compute the RMS brightness of the image at `image_path`.
    As outlined in the documentation:
      B_RMS = sqrt( mean( grayscale_value^2 ) )
    where grayscale_value = 0.299*R + 0.587*G + 0.114*B.
    
    Args:
        image_path (str): The file path to the image for which to compute brightness.
    
    Returns:
        float: The RMS brightness value of the image, or 0.0 if an error occurs.
    """
    try:
        # Attempt to load the image from the provided path using Pygame.
        surf = pygame.image.load(image_path)
        
        # Convert the loaded image to a 3D NumPy array of float32 type for numerical processing.
        arr = pygame.surfarray.array3d(surf).astype(np.float32)

        # Compute grayscale intensity for each pixel:
        #   gray = 0.299*R + 0.587*G + 0.114*B
        gray = (0.299 * arr[..., 0] +
                0.587 * arr[..., 1] +
                0.114 * arr[..., 2])

        # Final RMS brightness: sqrt( mean( gray^2 ) )
        return float(np.sqrt(np.mean(gray**2)))
    except Exception as e:
        logger.error(f"Error computing brightness for {image_path}: {e}")
        return 0.0


class GazeImageViewer:
    """
    Main application class for the image-based gaze-tracking experiment.
    
    This class initializes a Pygame environment, sets up the necessary views (ImageTaskView and
    ImageAnalysisView) and a GazeProcessor. It then connects to a Tobii eye tracker,
    manages the main application loop, handles user interactions, and controls the transitions
    between the “image task” phase and a post-task “analysis” phase.
    """

    def __init__(self, sort_by_brightness: bool = False):
        """
        Initialize the GazeImageViewer application.
        
        Args:
            sort_by_brightness (bool): Toggle whether to sort images by ascending RMS brightness.
                                       If False, images retain their alphabetical ordering.
        """
        logger.info("Initializing the GazeImageViewer application.")
        
        # Initialize all Pygame modules and set an application window caption.
        pygame.init()
        pygame.display.set_caption("Image Gaze Tracker")

        # Define the window dimensions; if desired, one could switch to fullscreen here.
        self.width = 1280
        self.height = 720
        self.screen = pygame.display.set_mode((self.width, self.height))

        # Initialize the two main views: image task (for real-time display) and analysis (for heatmaps).
        self.image_task_view = ImageTaskView(self.screen, self.width, self.height)
        self.analysis_view = ImageAnalysisView(self.screen, self.width, self.height)
        self.current_view = "image_task"  # Start with the image task phase.

        # Initialize a gaze processor to handle and smooth raw gaze data.
        self.gaze_processor = GazeProcessor(self.width, self.height)
        logger.debug("GazeProcessor initialized.")

        # Attempt to detect and initialize a Tobii eye tracker device.
        self.eyetracker = self.initialize_eyetracker()

        # Data containers for gaze points associated with each image.
        self.gaze_data_per_image = []  # List of lists, each sublist storing gaze data for a single image.
        self.current_image_data = []   # Temporary storage for the currently displayed image.

        # Flag to control brightness-based image sorting.
        self.sort_by_brightness = sort_by_brightness

        # Load images from the 'images' directory on startup.
        self.load_images()

    def initialize_eyetracker(self):
        """
        Discover and connect to a Tobii eye tracker.
        
        Returns:
            EyeTracker: The first Tobii eye tracker found.
        
        Raises:
            RuntimeError: If no Tobii device is detected.
        """
        logger.debug("Looking for connected eye trackers...")
        found_eyetrackers = tr.find_all_eyetrackers()
        if not found_eyetrackers:
            logger.error("No eye trackers found. Exiting.")
            raise RuntimeError("No eye trackers found.")
        logger.info(f"Found an eye tracker: {found_eyetrackers[0]}")
        return found_eyetrackers[0]

    def load_images(self) -> None:
        """
        Load images from the 'images' directory. If `sort_by_brightness` is True,
        the images will be sorted in ascending RMS brightness order; otherwise,
        they retain their original alphabetical ordering.
        
        Raises:
            RuntimeError: If the 'images' directory does not exist or contains no suitable files.
        """
        logger.info("Loading images from the 'images' directory.")
        image_dir = Path("images")

        # Check for existence of the images folder.
        if not image_dir.exists():
            logger.error("Images directory not found!")
            raise RuntimeError("Images directory not found.")

        # Collect all JPG or PNG image files from the directory.
        image_files = [
            str(f) for f in image_dir.iterdir()
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
        # Sort alphabetically by default.
        image_files.sort()

        # If brightness-based ordering is requested, re-sort using the RMS brightness measure.
        if self.sort_by_brightness:
            logger.debug("Sorting images by brightness (darkest to brightest).")
            image_files.sort(key=lambda path: compute_rms_brightness(path))

        # Ensure that we have at least one suitable image file.
        if not image_files:
            logger.warning("No suitable images found in 'images' directory.")
            raise RuntimeError("No suitable images found in 'images' directory.")

        logger.debug(f"Final image ordering: {image_files}")
        # Pass the final list of images to the ImageTaskView for display.
        self.image_task_view.set_images(image_files)

    def gaze_data_callback(self, gaze_data: dict) -> None:
        """
        Callback function invoked whenever new gaze data is received from the Tobii device.

        Args:
            gaze_data (dict): Dictionary typically containing keys like
                              'left_gaze_point_on_display_area' and
                              'right_gaze_point_on_display_area'.
        """
        # Only record gaze data if we are currently in the image task phase.
        if self.current_view == "image_task":
            self.gaze_processor.process_gaze_data(gaze_data)
            if self.gaze_processor.current_gaze:
                # Append the latest smoothed gaze position (int, int) to the current image's data list.
                self.current_image_data.append(self.gaze_processor.current_gaze)

    def handle_image_task_view(self) -> None:
        """
        Update the image task view, checking if the display time for the current image is complete
        and proceeding to the next image or transitioning to the analysis phase as needed.
        """
        # If the view is already marked complete, move on to analysis.
        if self.image_task_view.is_sequence_complete():
            logger.info("Sequence was already complete at the start of handle_image_task_view.")
            self.finish_task_and_switch_to_analysis()
            return

        # If the current image's duration has elapsed, store the gaze data and proceed.
        if self.image_task_view.check_image_complete():
            logger.debug(f"Completed image #{self.image_task_view.current_idx}. Storing gaze data.")
            self.gaze_data_per_image.append(self.current_image_data.copy())
            self.current_image_data.clear()
            self.image_task_view.next_image()

            # Check if we've now shown all images.
            if self.image_task_view.is_sequence_complete():
                logger.info("All images completed. Switching to analysis view.")
                self.finish_task_and_switch_to_analysis()
                return

        # Draw the current image along with any real-time gaze overlays.
        self.image_task_view.draw(
            current_gaze=self.gaze_processor.current_gaze,
            gaze_history=self.gaze_processor.gaze_history,
            remaining_images=self.image_task_view.remaining_images()
        )

    def finish_task_and_switch_to_analysis(self) -> None:
        """
        Once the last image in the sequence is displayed, stop recording gaze data
        and transition to the analysis view, which will display heatmaps
        """
        # Stop gaze capture to prevent additional data from being added mid-transition.
        self.gaze_processor.stop_recording()

        # Prepare the analysis phase by resetting the view and loading collected data.
        self.prepare_analysis()

        # Switch the current view to "analysis" mode.
        self.current_view = "analysis"

    def prepare_analysis(self) -> None:
        """
        Transfer image paths and associated gaze points to the analysis view,
        then initiate heatmap generation in a background thread.
        """
        logger.debug("Preparing analysis: resetting AnalysisView and loading data.")
        self.analysis_view.reset()
        self.analysis_view.load_images_and_data(
            self.image_task_view.image_paths,
            self.gaze_data_per_image
        )
        logger.debug("Starting heatmap generation in background.")
        self.analysis_view.start_heatmap_generation()

    def handle_analysis_view(self) -> None:
        """
        Render and update the analysis view, which may display heatmaps and
        allow navigation via arrow keys.
        """
        self.analysis_view.draw()

    def reset_session(self) -> None:
        """
        Reset the entire session, clearing out all stored gaze data
        and returning to the first phase of the experiment (the image task).
        """
        logger.info("Resetting session and returning to image task view.")
        self.gaze_data_per_image.clear()
        self.current_image_data.clear()
        self.gaze_processor.reset()
        self.image_task_view.reset()
        self.analysis_view.cleanup()
        self.current_view = "image_task"
        self.gaze_processor.start_recording()

    def cleanup(self) -> None:
        """
        Disconnect from the Tobii device, clean up resources, and quit Pygame.
        """
        logger.info("Cleaning up GazeImageViewer. Unsubscribing from Tobii, shutting down Pygame.")
        self.eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self.gaze_data_callback)
        self.analysis_view.cleanup()
        pygame.quit()

    def run(self) -> None:
        """
        Main application loop. Subscribes the gaze data callback, then processes Pygame events
        (QUIT, key presses, etc.) and updates whichever view (image task or analysis) is active.
        """
        logger.info("Starting main application loop.")
        # Subscribe to gaze data so the callback is triggered whenever fresh data arrives.
        self.eyetracker.subscribe_to(
            tr.EYETRACKER_GAZE_DATA,
            self.gaze_data_callback,
            as_dictionary=True
        )

        clock = pygame.time.Clock()
        running = True

        try:
            while running:
                # Run at up to 60 frames per second to keep UI rendering smooth.
                clock.tick(60)

                # Handle all pending Pygame events.
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        logger.info("Received QUIT event; exiting main loop.")
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        # Escape key closes the application.
                        if event.key == pygame.K_ESCAPE:
                            logger.info("ESC pressed; exiting main loop.")
                            running = False
                        # Space can reset the session if user is in the analysis phase.
                        elif event.key == pygame.K_SPACE and self.current_view == "analysis":
                            logger.debug("SPACE pressed in analysis view; resetting session.")
                            self.reset_session()
                        # Pressing 'g' toggles the gaze overlay during the image task phase.
                        elif event.key == pygame.K_g and self.current_view == "image_task":
                            logger.debug("Toggling gaze overlay in image task view.")
                            self.image_task_view.toggle_gaze_overlay()

                    if self.current_view == "analysis":
                        self.analysis_view.handle_input(event)

                # Decide which view should be updated/rendered based on current_view.
                if self.current_view == "image_task":
                    self.handle_image_task_view()
                else:
                    self.handle_analysis_view()

        finally:
            # If the loop ends (e.g., due to a QUIT event or error), clean up before exiting.
            logger.info("Cleanup.")
            self.cleanup()


if __name__ == "__main__":
    """
    Entry point for the GazeImageViewer application.
    
    Set sort_by_brightness=True to automatically sort the loaded images by ascending RMS brightness,
    so darkest images appear first. If False, images are processed in alphabetical order (the default).
    """
    # Example usage: 
    #   app = GazeImageViewer(sort_by_brightness=False)
    app = GazeImageViewer(sort_by_brightness=True)
    app.run()