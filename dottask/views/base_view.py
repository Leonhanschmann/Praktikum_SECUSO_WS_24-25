#views/base_view.py


# =========================================================================================
# BASE_VIEW.PY
#
# Defines a base view class for the gaze tracker application. This class
# acts as a foundational interface for drawing and managing display surfaces.
# It also provides utility methods for grid drawing, text rendering.
# =========================================================================================

import pygame
from typing import Tuple

class BaseView:
    """
    Base class for all views in the gaze tracker application.
    Provides common functionality for drawing and display management.
    """

    def __init__(self, screen: pygame.Surface, width: int, height: int):
        """
        Initialize the base view with common properties and surfaces.

        Args:
            screen (pygame.Surface): The pygame display surface (main window or sub-surface).
            width (int): The width of the screen in pixels.
            height (int): The height of the screen in pixels.
        """
        self.screen = screen
        self.width = width
        self.height = height
        
        # Initialize font for text rendering
        self.font = pygame.font.Font(None, 36)
        
        # Common color definitions
        self.BACKGROUND_COLOR = (0, 0, 0)   # Black background
        self.DOT_COLOR = (64, 196, 255)    # Light blue for drawing dots
        self.GAZE_COLOR = (64, 196, 255)   # Light blue for gaze visualization
        self.TEXT_COLOR = (255, 255, 255)  # White for text
        self.GRID_COLOR = (40, 40, 60)     # Dark blue-grey for grid lines
        
        # Create transparent surfaces for layered drawing (e.g., gaze dots vs. background)
        self.dot_surface = pygame.Surface((width, height), pygame.SRCALPHA, 32)
        self.gaze_surface = pygame.Surface((width, height), pygame.SRCALPHA, 32)
        self.dot_surface = self.dot_surface.convert_alpha()
        self.gaze_surface = self.gaze_surface.convert_alpha()

    def draw_aa_circle(self,
                       surface: pygame.Surface,
                       center: Tuple[int, int],
                       radius: int,
                       color: Tuple[int, int, int, int]) -> None:
        """
        Draw an anti-aliased circle by rendering multiple slightly offset circles
        to smooth out jagged edges.

        Args:
            surface (pygame.Surface): Surface to draw on.
            center (Tuple[int,int]): (x, y) coordinates of the circle center.
            radius (int): Circle radius in pixels.
            color (Tuple[int,int,int,int]): RGBA color tuple for the circle.
        """
        x, y = center
        for i in range(-2, 3):
            for j in range(-2, 3):
                pygame.draw.circle(
                    surface,
                    color,
                    (int(x + i * 0.25), int(y + j * 0.25)),
                    radius,
                    0
                )

    def draw_text(self,
                  text: str,
                  position: Tuple[int, int],
                  background_padding: int = 10) -> None:
        """
        Draw text with a semi-transparent background for better visibility.

        Args:
            text (str): The text string to display.
            position (Tuple[int,int]): (x, y) coordinates for the text position.
            background_padding (int): Extra space around the text background in pixels.
        """
        # Render text
        text_surface = self.font.render(text, True, self.TEXT_COLOR)
        text_rect = text_surface.get_rect(topleft=position)
        
        # Create and draw a semi-transparent background behind the text
        bg_rect = text_rect.inflate(background_padding, background_padding)
        bg_surface = pygame.Surface((bg_rect.width, bg_rect.height))
        bg_surface.fill((50, 50, 50))  # Dark grey background
        bg_surface.set_alpha(128)      # 50% transparency
        
        # Blit background and text onto the main screen
        self.screen.blit(bg_surface, bg_rect)
        self.screen.blit(text_surface, text_rect)

    def draw_grid(self, spacing: int = 50) -> None:
        """
        Draw a grid pattern on the screen to help visualize coordinates.

        Args:
            spacing (int): Distance in pixels between grid lines.
        """
        for x in range(0, self.width, spacing):
            pygame.draw.line(self.screen, self.GRID_COLOR, (x, 0), (x, self.height))
        for y in range(0, self.height, spacing):
            pygame.draw.line(self.screen, self.GRID_COLOR, (0, y), (self.width, y))

    def clear_surface(self, surface: pygame.Surface) -> None:
        """
        Clear a surface by filling it with transparent pixels.

        Args:
            surface (pygame.Surface): The surface to clear.
        """
        surface.fill((0, 0, 0, 0))

    def get_screen_dimensions(self) -> Tuple[int, int]:
        """
        Get the current screen dimensions.

        Returns:
            Tuple[int,int]: (width, height) of the screen.
        """
        return (self.width, self.height)

    def draw(self):
        """
        Override this method in subclasses to perform custom rendering.
        """
        pass

    def handle_event(self, e: pygame.event.Event):
        """
        Handle Pygame events. Subclasses should override if they need event handling.

        Args:
            e (pygame.event.Event): Event to handle.
        """
        pass

    def update(self, dt: float):
        """
        Update the view’s state or animations. Subclasses should override as needed.

        Args:
            dt (float): Time elapsed since the last update in seconds.
        """
        pass