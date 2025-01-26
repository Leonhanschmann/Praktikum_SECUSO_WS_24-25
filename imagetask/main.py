# =========================================================================================
# MAIN.PY
#
# Entry point for the GazeImageViewer application.
# Initializes the application, manages the main loop, handles events, and coordinates between
# different views and the gaze processor. Integrates with a Tobii eye-tracking device to
# capture real-time gaze data and visualize it over a sequence of images.
# =========================================================================================

import os
import time
import logging
from pathlib import Path

import pygame
import tobii_research as tr  # Requires Tobii SDK. 

import numpy as np
from PIL import Image  # Import Pillow library

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

def calculate_image_brightness(image_path: str) -> float:
    """
    Calculate the RMS brightness of an image using PIL and NumPy.

    Args:
        image_path (str): Path to the image file

    Returns:
        float: RMS brightness value of the image
    """
    try:
        # Open the image file
        with Image.open(image_path) as image:
            # Ensure the image is in RGB mode
            image = image.convert('RGB')
            # Convert image data to a NumPy array
            pixels = np.asarray(image)
            # Convert to grayscale using standard conversion weights
            grayscale = (
                0.299 * pixels[:, :, 0] + 
                0.587 * pixels[:, :, 1] + 
                0.114 * pixels[:, :, 2]
            )
            # Calculate RMS brightness
            rms = np.sqrt(np.mean(grayscale ** 2))
            return rms
    except Exception as e:
        logger.error(f"Failed to calculate brightness for {image_path}: {e}")
        raise

class GazeImageViewer:
    """
    Main application class for the image-based gaze tracking experiment.
    
    This class initializes the Pygame environment, sets up the necessary views and processors,
    connects to the Tobii eye-tracking device, manages the main application loop, handles user
    interactions, and orchestrates the transition between different phases of the experiment
    (image display and heatmap analysis).
    """

    def __init__(self):
        """
        Initialize the GazeImageViewer application.
        
        Sets up the Pygame window, initializes views, connects to the Tobii eye tracker, and
        prepares the image data for the experiment.
        """
        logger.info("Initializing the GazeImageViewer application.")
        pygame.init()  # Initialize all imported Pygame modules
        pygame.display.set_caption("Image Gaze Tracker")  # Set the window title

        # Define the window dimensions
        self.width = 1280
        self.height = 720
        self.screen = pygame.display.set_mode((self.width, self.height))  # Create the main display surface

        # Alternative fullscreen setup (commented out by default)
        # Uncomment the following lines to enable fullscreen mode
        # display_info = pygame.display.Info()
        # self.width = display_info.current_w
        # self.height = display_info.current_h
        # self.screen = pygame.display.set_mode(
        #     (self.width, self.height),
        #     pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF, 32
        # )

        # Initialize views
        self.image_task_view = ImageTaskView(self.screen, self.width, self.height)
        self.analysis_view = ImageAnalysisView(self.screen, self.width, self.height)
        self.current_view = "image_task"  # Set the initial view to the image task phase
        logger.debug("Set current view to 'image_task'")

        # Initialize the gaze processor with screen dimensions
        self.gaze_processor = GazeProcessor(self.width, self.height)
        logger.debug("GazeProcessor initialized")

        # Initialize and connect to the Tobii eye tracker
        self.eyetracker = self.initialize_eyetracker()

        # Data storage for gaze points per image
        self.gaze_data_per_image = []  # List to store gaze data for each image
        self.current_image_data = []   # Temporary storage for gaze data of the current image

        # Load images from the "images" directory
        self.load_images()

    def initialize_eyetracker(self):
        """
        Discover and initialize a connected Tobii eye tracker.
        
        Searches for connected Tobii eye trackers and connects to the first one found.
        If no eye trackers are found, logs an error and raises a RuntimeError.
        
        Returns:
            tobii_research.Eyetracker: The connected Tobii eye tracker instance.
        
        Raises:
            RuntimeError: If no Tobii eye trackers are found.
        """
        logger.debug("Looking for connected eye trackers...")
        found_eyetrackers = tr.find_all_eyetrackers()  # Discover all connected Tobii eye trackers
        if not found_eyetrackers:
            logger.error("No eye trackers found. Exiting.")
            raise RuntimeError("No eye trackers found.")
        logger.info(f"Found an eye tracker: {found_eyetrackers[0]}")
        return found_eyetrackers[0]  # Return the first found eye tracker

    def load_images(self) -> None:
        """
        Load and sort images from the "images" directory by their RMS brightness.
        
        Scans the "images" directory for image files with supported extensions, calculates their
        RMS brightness, sorts them accordingly, and passes the sorted list to the image task view.
        If no suitable images are found, logs a warning and raises a RuntimeError.
        
        Raises:
            RuntimeError: If the "images" directory does not exist or contains no suitable images.
        """
        logger.info("Loading images from the 'images' directory.")
        image_dir = Path("images")  # Define the path to the images directory
        if not image_dir.exists() or not image_dir.is_dir():
            logger.error("Images directory not found!")
            raise RuntimeError("Images directory not found.")

        # Collect image files with supported extensions
        supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
        image_files = [
            f for f in image_dir.iterdir()
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]

        if not image_files:
            logger.warning("No suitable images found in 'images' directory!")
            raise RuntimeError("No suitable images found in 'images' directory.")

        # Calculate brightness for each image and create (path, brightness) pairs
        image_brightness = []
        for img_path in image_files:
            try:
                brightness = calculate_image_brightness(str(img_path))
                image_brightness.append((str(img_path), brightness))
                logger.debug(f"Image {img_path}: brightness = {brightness}")
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue

        if not image_brightness:
            logger.error("No images were successfully processed.")
            raise RuntimeError("No images were successfully processed.")

        # Sort images by brightness (darker images first)
        sorted_images = [
            img_path for img_path, _ in sorted(image_brightness, key=lambda x: x[1])
        ]

        logger.info(f"Found {len(sorted_images)} images, sorted by brightness.")
        # Pass the sorted list of image paths to the image task view
        self.image_task_view.set_images(sorted_images)

    def gaze_data_callback(self, gaze_data: dict) -> None:
        """
        Callback function to handle incoming gaze data from the eye tracker.
        
        This method is called whenever new gaze data is received from the Tobii eye tracker.
        If the current view is the image task phase, it processes the gaze data using the
        gaze processor and appends the current gaze position to the image's gaze data list.
        
        Args:
            gaze_data (dict): Dictionary containing gaze data from the eye tracker.
                Expected keys:
                    'left_gaze_point_on_display_area': (lx, ly),
                    'right_gaze_point_on_display_area': (rx, ry)
        """
        if self.current_view == "image_task":
            # Process the incoming gaze data
            self.gaze_processor.process_gaze_data(gaze_data)
            if self.gaze_processor.current_gaze:
                # Append the current gaze position to the image's gaze data list
                self.current_image_data.append(self.gaze_processor.current_gaze)

    def handle_image_task_view(self) -> None:
        """
        Update and render the image task view.
        
        This method checks if the current image's display time has elapsed and moves to the
        next image if necessary. If all images have been displayed, it transitions to the
        analysis view. Otherwise, it continues to render the current image with gaze visualization.
        """
        # Check if the image sequence is already complete
        if self.image_task_view.is_sequence_complete():
            logger.info("Sequence was already complete at the start of handle_image_task_view.")
            self.finish_task_and_switch_to_analysis()
            return

        # Check if the current image's display time has elapsed
        if self.image_task_view.check_image_complete():
            logger.debug(f"Completed image #{self.image_task_view.current_idx}. Storing gaze data.")
            # Store the accumulated gaze data for the completed image
            self.gaze_data_per_image.append(self.current_image_data.copy())
            self.current_image_data.clear()  # Clear temporary storage for the next image

            # Advance to the next image in the sequence
            self.image_task_view.next_image()

            # Check again if the sequence is complete after moving to the next image
            if self.image_task_view.is_sequence_complete():
                logger.info("All images completed. Switching to analysis view.")
                self.finish_task_and_switch_to_analysis()
                return

        # Render the current image task view with gaze visualization
        self.image_task_view.draw(
            current_gaze=self.gaze_processor.current_gaze,
            gaze_history=self.gaze_processor.gaze_history,
            remaining_images=self.image_task_view.remaining_images()
        )

    def finish_task_and_switch_to_analysis(self) -> None:
        """
        Finalize the image task phase and transition to the analysis view.
        
        This method stops the gaze processor from recording further gaze data, prepares the
        analysis view by passing the collected gaze data and image paths, and updates the
        current view to the analysis phase.
        """
        # Stop recording gaze data
        self.gaze_processor.stop_recording()
        # Prepare the analysis view with the collected data
        self.prepare_analysis()
        # Switch the current view to the analysis phase
        self.current_view = "analysis"

    def prepare_analysis(self) -> None:
        """
        Prepare the analysis view with the necessary data for heatmap generation.
        
        This method resets the analysis view, loads the images and corresponding gaze data,
        and initiates the background heatmap generation process.
        """
        logger.debug("Preparing analysis: resetting analysis view and loading image data.")
        self.analysis_view.reset()  # Reset the analysis view to clear any previous state
        self.analysis_view.load_images_and_data(
            self.image_task_view.image_paths,
            self.gaze_data_per_image
        )
        # Start generating heatmaps in the background
        logger.debug("Starting heatmap generation in background.")
        self.analysis_view.start_heatmap_generation()

    def handle_analysis_view(self) -> None:
        """
        Render the analysis view.
        
        This method delegates the drawing process to the analysis view, which handles
        displaying heatmaps, loading screens, and user navigation between heatmaps.
        """
        self.analysis_view.draw()

    def reset_session(self) -> None:
        """
        Reset the entire session, clearing all stored data and returning to the image task phase.
        
        This method clears accumulated gaze data, resets the gaze processor and image task view,
        cleans up the analysis view, and reactivates gaze recording. It effectively restarts
        the experiment for a new run.
        """
        logger.info("Resetting session and returning to image task view.")
        self.gaze_data_per_image.clear()   # Clear all stored gaze data per image
        self.current_image_data.clear()    # Clear temporary gaze data for the current image

        self.gaze_processor.reset()        # Reset the gaze processor to clear its state
        self.image_task_view.reset()       # Reset the image task view to start from the first image
        self.analysis_view.cleanup()       # Clean up the analysis view resources

        self.current_view = "image_task"    # Switch back to the image task view
        self.gaze_processor.start_recording()  # Reactivate gaze recording

    def cleanup(self) -> None:
        """
        Clean up resources and gracefully shut down the application.
        
        This method unsubscribes from the Tobii eye tracker, cleans up the analysis view,
        and quits Pygame to ensure all resources are properly released.
        """
        logger.info("Cleaning up GazeImageViewer. Unsubscribing from Tobii, shutting down Pygame.")
        # Unsubscribe the gaze data callback from the eye tracker to stop receiving data
        self.eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self.gaze_data_callback)
        # Clean up the analysis view to release any allocated resources
        self.analysis_view.cleanup()
        # Quit Pygame to close the display and terminate all Pygame modules
        pygame.quit()

    def run(self) -> None:
        """
        Run the main application loop.
        
        This method subscribes to the Tobii eye tracker to start receiving gaze data,
        enters the main loop where it handles events, updates views, and renders content.
        The loop continues until a quit event is received, at which point it cleans up
        resources and exits.
        """
        logger.info("Starting main application loop.")
        # Subscribe the gaze data callback to the eye tracker to start receiving data
        self.eyetracker.subscribe_to(
            tr.EYETRACKER_GAZE_DATA, self.gaze_data_callback, as_dictionary=True
        )

        try:
            clock = pygame.time.Clock()  # Create a clock to manage the frame rate
            running = True  # Flag to control the main loop

            while running:
                clock.tick(60)  # Limit the loop to 60 frames per second
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        # Handle the window close event
                        logger.info("Received QUIT event; exiting main loop.")
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            # Allow the user to exit the application by pressing ESC
                            logger.info("ESC pressed; exiting main loop.")
                            running = False
                        elif (event.key == pygame.K_SPACE and self.current_view == "analysis"):
                            # Allow the user to reset the session by pressing SPACE in the analysis view
                            logger.debug("SPACE pressed in analysis view; resetting session.")
                            self.reset_session()
                        elif (event.key == pygame.K_g and self.current_view == "image_task"):
                            # Allow the user to toggle gaze overlay by pressing 'g' in the image task view
                            logger.debug("Toggling gaze overlay in image task view.")
                            self.image_task_view.toggle_gaze_overlay()

                    if self.current_view == "analysis":
                        # Delegate event handling to the analysis view for navigation
                        self.analysis_view.handle_input(event)

                # Update and render the appropriate view based on the current phase
                if self.current_view == "image_task":
                    self.handle_image_task_view()
                else:
                    self.handle_analysis_view()

        finally:
            # Ensure that cleanup is performed regardless of how the loop exits
            logger.info("Cleanup.")
            self.cleanup()


if __name__ == "__main__":
    """
    Entry point for the GazeImageViewer application.
    
    Creates an instance of GazeImageViewer and starts the main application loop.
    """
    app = GazeImageViewer()
    app.run()