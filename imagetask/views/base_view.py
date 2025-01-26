# =========================================================================================
# BASE_VIEW.PY
#
# Base class for all views in the image task application.
# Provides common functionalities such as drawing utilities, text rendering,
# grid overlays, and event handling that can be extended by specific view implementations.
# =========================================================================================

import pygame
from typing import Tuple

class BaseView:
    """
    Base class for all views in the gaze tracker application.
    
    This class encapsulates common functionalities required by different views,
    such as drawing anti-aliased circles, rendering text, drawing grids, and managing
    surfaces. It serves as an abstract foundation that other specific views can inherit
    and extend with additional behaviors.
    """

    def __init__(self, screen: pygame.Surface, width: int, height: int):
        """
        Initialize the BaseView with essential properties and surfaces.
        
        Args:
            screen (pygame.Surface): The main display surface where all drawings occur.
            width (int): Width of the display in pixels.
            height (int): Height of the display in pixels.
        """
        self.screen = screen
        self.width = width
        self.height = height
        
        # Initialize font for text rendering; default font with size 36
        self.font = pygame.font.Font(None, 36)
        
        # Define common colors used across different views
        self.BACKGROUND_COLOR = (0, 0, 0)        # Black background
        self.DOT_COLOR = (64, 196, 255)          # Light blue color for dots
        self.GAZE_COLOR = (64, 196, 255)         # Light blue color for gaze visualization
        self.TEXT_COLOR = (255, 255, 255)        # White color for text
        self.GRID_COLOR = (40, 40, 60)           # Darker color for grid lines
        
        # Create separate surfaces for drawing dots and gaze visuals with per-pixel alpha
        self.dot_surface = pygame.Surface((width, height), pygame.SRCALPHA, 32).convert_alpha()
        self.gaze_surface = pygame.Surface((width, height), pygame.SRCALPHA, 32).convert_alpha()

    def draw_aa_circle(
        self,
        surface: pygame.Surface,
        center: Tuple[int, int],
        radius: int,
        color: Tuple[int, int, int, int]
    ) -> None:
        """
        Draw an anti-aliased circle on the specified surface.
        
        Since Pygame does not natively support anti-aliased circles, this method
        approximates anti-aliasing by drawing multiple smaller circles around the
        intended position to smooth out the edges.
        
        Args:
            surface (pygame.Surface): The surface on which to draw the circle.
            center (Tuple[int, int]): The (x, y) coordinates of the circle's center.
            radius (int): The radius of the circle in pixels.
            color (Tuple[int, int, int, int]): The RGBA color of the circle.
        """
        x, y = center
        # Iterate over a small grid around the center to create an anti-aliased effect
        for i in range(-2, 3):
            for j in range(-2, 3):
                # Draw smaller circles slightly offset from the center
                pygame.draw.circle(
                    surface,
                    color,
                    (int(x + i * 0.25), int(y + j * 0.25)),
                    radius,
                    0  # Filled circle
                )

    def draw_text(
        self,
        text: str,
        position: Tuple[int, int],
        background_padding: int = 10
    ) -> None:
        """
        Render and draw text on the screen with an optional semi-transparent background.
        
        This method helps in making the text readable against various backgrounds by
        providing a padded, semi-transparent rectangle behind the text.
        
        Args:
            text (str): The text string to render.
            position (Tuple[int, int]): The (x, y) coordinates where the text will be placed.
            background_padding (int, optional): Padding around the text background. Defaults to 10.
        """
        # Render the text surface with the specified text and color
        text_surface = self.font.render(text, True, self.TEXT_COLOR)
        text_rect = text_surface.get_rect(topleft=position)
        
        # Inflate the text rectangle to create a background padding
        bg_rect = text_rect.inflate(background_padding, background_padding)
        # Create a new surface for the background with the inflated size
        bg_surface = pygame.Surface((bg_rect.width, bg_rect.height))
        bg_surface.fill((50, 50, 50))  # Dark gray background
        bg_surface.set_alpha(128)       # Semi-transparent
        
        # Blit the background rectangle and then the text on top
        self.screen.blit(bg_surface, bg_rect)
        self.screen.blit(text_surface, text_rect)

    def draw_grid(self, spacing: int = 50) -> None:
        """
        Draw a grid overlay on the screen for reference.
        
        The grid is composed of vertical and horizontal lines spaced at regular intervals.
        
        Args:
            spacing (int, optional): The number of pixels between grid lines. Defaults to 50.
        """
        # Draw vertical grid lines
        for x in range(0, self.width, spacing):
            pygame.draw.line(self.screen, self.GRID_COLOR, (x, 0), (x, self.height))
        # Draw horizontal grid lines
        for y in range(0, self.height, spacing):
            pygame.draw.line(self.screen, self.GRID_COLOR, (0, y), (self.width, y))

    def clear_surface(self, surface: pygame.Surface) -> None:
        """
        Clear the specified surface by filling it with a fully transparent color.
        
        This is useful for resetting temporary surfaces before redrawing new content.
        
        Args:
            surface (pygame.Surface): The surface to clear.
        """
        surface.fill((0, 0, 0, 0))  # Fully transparent

    def get_screen_dimensions(self) -> Tuple[int, int]:
        """
        Retrieve the current dimensions of the screen.
        
        Returns:
            Tuple[int, int]: A tuple containing the width and height of the screen.
        """
        return (self.width, self.height)

    def draw(self):
        """
        Placeholder method for drawing content.
        
        This method is intended to be overridden by subclasses to implement specific
        drawing logic for different views.
        """
        pass

    def handle_event(self, e: pygame.event.Event):
        """
        Placeholder method for handling events.
        
        This method is intended to be overridden by subclasses to handle specific
        events relevant to different views.
        
        Args:
            e (pygame.event.Event): The event to handle.
        """
        pass

    def update(self, dt: float):
        """
        Placeholder method for updating the view's state.
        
        This method is intended to be overridden by subclasses to implement specific
        update logic, such as animations or state changes, based on the elapsed time.
        
        Args:
            dt (float): The elapsed time in seconds since the last update.
        """
        pass
