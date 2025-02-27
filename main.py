import pygame
import numpy as np
import random
import math
import time
import sys
import os
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1000, 700
GRID_SIZE = 800
GRID_CELLS = 40
CELL_SIZE = GRID_SIZE // GRID_CELLS
SIDEBAR_WIDTH = WIDTH - GRID_SIZE
INFO_HEIGHT = 200

# Color Palette (Modern & Professional)
BLACK = (20, 20, 20)
WHITE = (255, 255, 255)
LIGHT_GRAY = (240, 240, 240)
GRAY = (190, 190, 190)
DARK_GRAY = (80, 80, 80)
BG_COLOR = (245, 247, 250)  # Light blue-gray background

# Primary Colors
PRIMARY = (67, 99, 216)  # Blue
PRIMARY_DARK = (46, 64, 146)
PRIMARY_LIGHT = (156, 175, 241)

# Secondary Colors
SECONDARY = (79, 195, 247)  # Light blue
SECONDARY_LIGHT = (170, 224, 255)

# Accent Colors
ACCENT_SUCCESS = (72, 187, 120)  # Green
ACCENT_DANGER = (220, 53, 69)  # Red
ACCENT_WARNING = (255, 193, 7)  # Yellow
ACCENT_INFO = (23, 162, 184)  # Teal
ACCENT_NEUTRAL = (108, 117, 125)  # Gray

# Ant & Path Colors
ANT_COLOR = (165, 42, 42)  # Dark brown
PHEROMONE_COLOR = (90, 120, 255, 128)  # Semi-transparent blue
BEST_PATH_COLOR = (72, 187, 120)  # Green
START_COLOR = (72, 187, 120)  # Green
END_COLOR = (220, 53, 69)  # Red

# UI Colors
BUTTON_COLOR = PRIMARY
BUTTON_HOVER = PRIMARY_DARK
BUTTON_TEXT = WHITE
PANEL_BG = WHITE
SLIDER_TRACK = GRAY
SLIDER_HANDLE = PRIMARY

# Fonts
pygame.font.init()
try:
    # Try to use more modern fonts if available
    FONT_SMALL = pygame.font.SysFont('Segoe UI', 14)
    FONT_MEDIUM = pygame.font.SysFont('Segoe UI', 16)
    FONT_LARGE = pygame.font.SysFont('Segoe UI', 20)
    FONT_TITLE = pygame.font.SysFont('Segoe UI', 28, bold=True)
except:
    # Fallback to basic fonts
    FONT_SMALL = pygame.font.SysFont('Arial', 14)
    FONT_MEDIUM = pygame.font.SysFont('Arial', 16)
    FONT_LARGE = pygame.font.SysFont('Arial', 20)
    FONT_TITLE = pygame.font.SysFont('Arial', 28, bold=True)


class AntColonyOptimization:
    def __init__(self, grid_size: int, n_ants: int = 20, alpha: float = 1.0, beta: float = 2.0,
                 evaporation_rate: float = 0.5, q: float = 100.0, initial_pheromone: float = 0.1):
        """
        Initialize the Ant Colony Optimization algorithm

        Args:
            grid_size: Size of the grid (grid_size x grid_size)
            n_ants: Number of ants in the colony
            alpha: Pheromone importance factor
            beta: Heuristic information importance factor
            evaporation_rate: Rate at which pheromone evaporates
            q: Pheromone deposit factor
            initial_pheromone: Initial pheromone value
        """
        self.grid_size = grid_size
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q = q

        # Initialize grid (0 = empty, 1 = obstacle, 2 = start, 3 = end)
        self.grid = np.zeros((grid_size, grid_size), dtype=int)

        # Pheromone matrix
        self.pheromones = np.ones((grid_size, grid_size)) * initial_pheromone

        # Start and end positions
        self.start_pos = (grid_size // 5, grid_size // 2)
        self.end_pos = (grid_size - grid_size // 5, grid_size // 2)
        self.grid[self.start_pos] = 2
        self.grid[self.end_pos] = 3

        # Ants
        self.ants = []
        self.reset_ants()

        # Stats
        self.best_path_length = float('inf')
        self.best_path = []
        self.avg_path_lengths = []
        self.best_path_lengths = []
        self.iterations = 0
        self.convergence_count = 0
        self.converged = False

        # Directions (up, right, down, left, diagonals)
        self.directions = [
            (-1, 0), (0, 1), (1, 0), (0, -1),
            (-1, -1), (-1, 1), (1, 1), (1, -1)
        ]

    def reset_ants(self):
        """Reset all ants to the starting position"""
        self.ants = []
        for _ in range(self.n_ants):
            self.ants.append({
                'position': self.start_pos,
                'path': [self.start_pos],
                'visited': set([self.start_pos]),
                'done': False
            })

    def heuristic(self, pos: Tuple[int, int]) -> float:
        """Calculate heuristic value (distance to end)"""
        return math.sqrt((pos[0] - self.end_pos[0]) ** 2 + (pos[1] - self.end_pos[1]) ** 2)

    def get_valid_moves(self, ant: Dict) -> List[Tuple[int, int]]:
        """Get valid moves for an ant"""
        valid_moves = []
        current_pos = ant['position']

        for dx, dy in self.directions:
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)

            # Check if the move is within grid bounds
            if 0 <= next_pos[0] < self.grid_size and 0 <= next_pos[1] < self.grid_size:
                # Check if not an obstacle and not visited
                if self.grid[next_pos] != 1 and next_pos not in ant['visited']:
                    valid_moves.append(next_pos)

        return valid_moves

    def choose_next_move(self, ant: Dict) -> Tuple[int, int]:
        """Choose the next move for an ant using ACO probabilities"""
        valid_moves = self.get_valid_moves(ant)

        if not valid_moves:
            # No valid moves, ant is stuck
            return None

        # Calculate probabilities for each move
        probabilities = []
        for move in valid_moves:
            # Pheromone value
            tau = self.pheromones[move]

            # Heuristic value (inverse of distance to end)
            eta = 1.0 / (self.heuristic(move) + 0.1)  # Add 0.1 to avoid division by zero

            # Calculate probability
            probability = (tau ** self.alpha) * (eta ** self.beta)
            probabilities.append(probability)

        # Normalize probabilities
        total = sum(probabilities)
        if total == 0:
            # If all probabilities are zero, choose randomly
            return random.choice(valid_moves)

        probabilities = [p / total for p in probabilities]

        # Choose move based on probabilities
        return np.random.choice(len(valid_moves), p=probabilities)

    def move_ants(self) -> List[Dict]:
        """Move all ants one step forward"""
        ants_at_end = []

        for ant in self.ants:
            if ant['done']:
                continue

            valid_moves = self.get_valid_moves(ant)

            if not valid_moves:
                # No valid moves, ant is stuck
                ant['done'] = True
                continue

            # Calculate probabilities for each move
            probabilities = []
            for move in valid_moves:
                # Pheromone value
                tau = self.pheromones[move]

                # Heuristic value (inverse of distance to end)
                eta = 1.0 / (self.heuristic(move) + 0.1)  # Add 0.1 to avoid division by zero

                # Calculate probability
                probability = (tau ** self.alpha) * (eta ** self.beta)
                probabilities.append(probability)

            # Normalize probabilities
            total = sum(probabilities)
            if total == 0:
                # If all probabilities are zero, choose randomly
                next_pos = random.choice(valid_moves)
            else:
                probabilities = [p / total for p in probabilities]
                next_pos_idx = np.random.choice(len(valid_moves), p=probabilities)
                next_pos = valid_moves[next_pos_idx]

            # Move ant
            ant['position'] = next_pos
            ant['path'].append(next_pos)
            ant['visited'].add(next_pos)

            # Check if ant reached the end
            if next_pos == self.end_pos:
                ant['done'] = True
                ants_at_end.append(ant)

        return ants_at_end

    def update_pheromones(self, ants_at_end: List[Dict]):
        """Update pheromone levels based on ant paths"""
        # Evaporation
        self.pheromones *= (1 - self.evaporation_rate)

        # Deposit new pheromones
        for ant in ants_at_end:
            path_length = len(ant['path']) - 1  # -1 because we count edges, not nodes
            if path_length < 1:
                continue

            # Calculate pheromone deposit
            deposit = self.q / path_length

            # Update pheromones on the path
            for pos in ant['path']:
                self.pheromones[pos] += deposit

            # Update best path
            if path_length < self.best_path_length:
                self.best_path_length = path_length
                self.best_path = ant['path'].copy()

    def run_iteration(self) -> bool:
        """Run one iteration of the algorithm"""
        self.reset_ants()
        self.iterations += 1

        # Move ants until all are done
        all_done = False
        steps = 0
        max_steps = self.grid_size * 2  # Limit to avoid infinite loops

        ants_at_end = []

        while not all_done and steps < max_steps:
            new_ants_at_end = self.move_ants()
            ants_at_end.extend(new_ants_at_end)

            # Check if all ants are done
            all_done = all(ant['done'] for ant in self.ants)
            steps += 1

        # Update pheromones
        self.update_pheromones(ants_at_end)

        # Calculate statistics
        path_lengths = [len(ant['path']) - 1 for ant in ants_at_end] if ants_at_end else [float('inf')]
        avg_path_length = sum(path_lengths) / len(path_lengths) if path_lengths else float('inf')

        self.avg_path_lengths.append(avg_path_length if avg_path_length != float('inf') else None)
        self.best_path_lengths.append(self.best_path_length if self.best_path_length != float('inf') else None)

        # Check for convergence (if best path has not improved for several iterations)
        if len(self.best_path_lengths) > 10 and self.best_path_lengths[-1] == self.best_path_lengths[-10]:
            self.convergence_count += 1
        else:
            self.convergence_count = 0

        if self.convergence_count >= 10:
            self.converged = True

        return self.converged

    def add_obstacle(self, pos: Tuple[int, int]):
        """Add an obstacle to the grid"""
        if pos != self.start_pos and pos != self.end_pos:
            self.grid[pos] = 1
            # Reset best path if obstacle added
            if self.best_path and pos in self.best_path:
                self.best_path = []
                self.best_path_length = float('inf')

    def remove_obstacle(self, pos: Tuple[int, int]):
        """Remove an obstacle from the grid"""
        if self.grid[pos] == 1:
            self.grid[pos] = 0

    def clear_obstacles(self):
        """Clear all obstacles from the grid"""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == 1:
                    self.grid[i, j] = 0

        # Reset best path
        self.best_path = []
        self.best_path_length = float('inf')

    def reset(self):
        """Reset the algorithm"""
        # Keep obstacles, but reset pheromones and stats
        self.pheromones = np.ones((self.grid_size, self.grid_size)) * 0.1
        self.best_path_length = float('inf')
        self.best_path = []
        self.avg_path_lengths = []
        self.best_path_lengths = []
        self.iterations = 0
        self.convergence_count = 0
        self.converged = False
        self.reset_ants()

    def set_parameters(self, n_ants: int, alpha: float, beta: float, evaporation_rate: float):
        """Set algorithm parameters"""
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.reset_ants()


class Button:
    def __init__(self, x: int, y: int, width: int, height: int, text: str,
                 color: Tuple[int, int, int] = BUTTON_COLOR,
                 hover_color: Tuple[int, int, int] = BUTTON_HOVER,
                 text_color: Tuple[int, int, int] = BUTTON_TEXT,
                 font=FONT_MEDIUM,
                 icon=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.font = font
        self.hovered = False
        self.clicked = False
        self.icon = icon
        self.animation_progress = 0

    def draw(self, screen):
        # Base button with shadow effect
        shadow_rect = pygame.Rect(self.rect.x + 2, self.rect.y + 2, self.rect.width, self.rect.height)
        pygame.draw.rect(screen, DARK_GRAY, shadow_rect, border_radius=8)

        # Determine button color based on state
        if self.clicked:
            color = self.hover_color
            # Move button slightly down to give pressed effect
            button_rect = pygame.Rect(self.rect.x + 1, self.rect.y + 1, self.rect.width - 1, self.rect.height - 1)
        else:
            color = self.hover_color if self.hovered else self.color
            button_rect = self.rect

        # Draw button with gradient effect
        pygame.draw.rect(screen, color, button_rect, border_radius=8)

        # Add subtle highlight on top
        highlight_rect = pygame.Rect(button_rect.x, button_rect.y, button_rect.width, 4)
        highlight_color = tuple(min(c + 30, 255) for c in color)
        pygame.draw.rect(screen, highlight_color, highlight_rect, border_radius=8)

        # Draw text with slight shadow for better readability
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=button_rect.center)

        # Draw text shadow
        shadow_surface = self.font.render(self.text, True, (50, 50, 50))
        shadow_rect = shadow_surface.get_rect(center=(text_rect.centerx + 1, text_rect.centery + 1))
        screen.blit(shadow_surface, shadow_rect)

        # Draw actual text
        screen.blit(text_surface, text_rect)

        # Draw icon if provided
        if self.icon:
            pass  # Icon implementation could be added here

        # Reset click animation
        if self.clicked:
            self.clicked = False

    def check_hover(self, pos: Tuple[int, int]):
        self.hovered = self.rect.collidepoint(pos)
        return self.hovered

    def check_click(self, pos: Tuple[int, int]):
        if self.rect.collidepoint(pos):
            self.clicked = True
            return True
        return False


class Slider:
    def __init__(self, x: int, y: int, width: int, height: int, min_val: float, max_val: float,
                 initial_val: float, text: str, format_str: str = "{:.1f}",
                 color=SLIDER_TRACK, handle_color=SLIDER_HANDLE):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.text = text
        self.format_str = format_str
        self.color = color
        self.handle_color = handle_color
        self.dragging = False
        self.handle_rect = pygame.Rect(0, 0, 18, height + 12)
        self.active_area = pygame.Rect(x - 10, y - 10, width + 20, height + 20)
        self.hovered = False
        self.update_handle()

    def update_handle(self):
        value_pos = ((self.value - self.min_val) / (self.max_val - self.min_val)) * self.rect.width
        self.handle_rect.centerx = self.rect.x + value_pos
        self.handle_rect.centery = self.rect.centery

    def draw(self, screen):
        # Draw label with shadow for better readability
        shadow_surface = FONT_MEDIUM.render(self.text, True, DARK_GRAY)
        text_surface = FONT_MEDIUM.render(self.text, True, BLACK)
        screen.blit(shadow_surface, (self.rect.x + 1, self.rect.y - 25))
        screen.blit(text_surface, (self.rect.x, self.rect.y - 26))

        # Draw filled progress portion
        progress_width = ((self.value - self.min_val) / (self.max_val - self.min_val)) * self.rect.width
        progress_rect = pygame.Rect(self.rect.x, self.rect.y, progress_width, self.rect.height)

        # Draw track (background) with rounded corners
        pygame.draw.rect(screen, LIGHT_GRAY, self.rect, border_radius=4)
        pygame.draw.rect(screen, self.color, progress_rect, border_radius=4)

        # Draw handle with shadow and highlight effects
        if self.dragging:
            # Shadow underneath
            shadow_rect = pygame.Rect(
                self.handle_rect.x + 1,
                self.handle_rect.y + 1,
                self.handle_rect.width,
                self.handle_rect.height
            )
            pygame.draw.rect(screen, DARK_GRAY, shadow_rect, border_radius=9)

            # Handle
            handle_color = tuple(max(0, min(255, c - 20)) for c in self.handle_color)
            pygame.draw.rect(screen, handle_color, self.handle_rect, border_radius=9)

            # Highlight on top
            highlight_rect = pygame.Rect(
                self.handle_rect.x,
                self.handle_rect.y,
                self.handle_rect.width,
                5
            )
            highlight_color = tuple(min(c + 30, 255) for c in handle_color)
            pygame.draw.rect(screen, highlight_color, highlight_rect, border_radius=9)
        else:
            # Shadow underneath when not dragging
            shadow_rect = pygame.Rect(
                self.handle_rect.x + 2,
                self.handle_rect.y + 2,
                self.handle_rect.width,
                self.handle_rect.height
            )
            pygame.draw.rect(screen, DARK_GRAY, shadow_rect, border_radius=9)

            # Handle
            handle_color = tuple(min(c + 20, 255) for c in self.handle_color) if self.hovered else self.handle_color
            pygame.draw.rect(screen, handle_color, self.handle_rect, border_radius=9)

            # Highlight on top
            highlight_rect = pygame.Rect(
                self.handle_rect.x,
                self.handle_rect.y,
                self.handle_rect.width,
                5
            )
            highlight_color = tuple(min(c + 40, 255) for c in handle_color)
            pygame.draw.rect(screen, highlight_color, highlight_rect, border_radius=9)

        # Draw value with nice background
        value_text = self.format_str.format(self.value)
        value_surface = FONT_MEDIUM.render(value_text, True, BLACK)
        value_width = value_surface.get_width() + 10
        value_bg_rect = pygame.Rect(self.rect.right + 15, self.rect.centery - 12, value_width, 24)

        # Value background
        pygame.draw.rect(screen, LIGHT_GRAY, value_bg_rect, border_radius=5)

        # Value text
        screen.blit(value_surface, (self.rect.right + 20, self.rect.centery - 10))

    def check_hover(self, pos: Tuple[int, int]):
        self.hovered = self.handle_rect.collidepoint(pos)
        return self.hovered

    def check_click(self, pos: Tuple[int, int]):
        if self.handle_rect.collidepoint(pos):
            self.dragging = True
            return True
        elif self.rect.collidepoint(pos):
            # Allow clicking anywhere on the track to jump
            rel_x = max(0, min(pos[0] - self.rect.x, self.rect.width))
            self.value = self.min_val + (rel_x / self.rect.width) * (self.max_val - self.min_val)
            self.update_handle()
            self.dragging = True
            return True
        return False

    def update(self, pos: Tuple[int, int]):
        if self.dragging:
            rel_x = max(0, min(pos[0] - self.rect.x, self.rect.width))
            self.value = self.min_val + (rel_x / self.rect.width) * (self.max_val - self.min_val)
            self.update_handle()

    def stop_drag(self):
        self.dragging = False


def create_stats_chart(aco: AntColonyOptimization) -> pygame.Surface:
    """Create a chart showing algorithm performance over time"""
    # Create matplotlib figure
    fig = Figure(figsize=(5, 3), dpi=80)
    ax = fig.add_subplot(111)

    # Filter out None values
    iterations = list(range(1, len(aco.best_path_lengths) + 1))
    best_paths = [p for p in aco.best_path_lengths if p is not None]
    best_iterations = [iterations[i] for i, p in enumerate(aco.best_path_lengths) if p is not None]

    avg_paths = [p for p in aco.avg_path_lengths if p is not None]
    avg_iterations = [iterations[i] for i, p in enumerate(aco.avg_path_lengths) if p is not None]

    # Plot data
    if best_paths:
        ax.plot(best_iterations, best_paths, 'b-', label='Best Path')
    if avg_paths:
        ax.plot(avg_iterations, avg_paths, 'r-', label='Avg Path')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Path Length')
    ax.set_title('Path Optimization Progress')
    ax.grid(True, linestyle='--', alpha=0.7)

    if best_paths or avg_paths:
        ax.legend()

    # Convert matplotlib figure to pygame surface
    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    # Use a simpler method for conversion to avoid dimension issues
    width, height = canvas.get_width_height()
    chart_surf = pygame.Surface((width, height))
    chart_rgb_array = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    chart_rgb_array = chart_rgb_array.reshape((height, width, 4))

    # Create a temporary pygame surface from the data
    temp_surf = pygame.image.frombuffer(chart_rgb_array.tobytes(), (width, height), 'RGBA')
    chart_surf.blit(temp_surf, (0, 0))

    return chart_surf


def draw_grid(screen, aco: AntColonyOptimization, draw_pheromones: bool = True):
    """Draw the grid with obstacles, start, end, and ants"""
    # Draw grid background with subtle patterns
    grid_bg = pygame.Surface((GRID_SIZE, GRID_SIZE))
    grid_bg.fill(BG_COLOR)

    # Create a subtle grid pattern
    for i in range(GRID_CELLS):
        for j in range(GRID_CELLS):
            if (i + j) % 2 == 0:
                cell_rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(grid_bg, (BG_COLOR[0] - 5, BG_COLOR[1] - 5, BG_COLOR[2] - 5), cell_rect)

    screen.blit(grid_bg, (0, 0))

    # Draw grid border
    pygame.draw.rect(screen, DARK_GRAY, (0, 0, GRID_SIZE, GRID_SIZE), 2)

    # Draw grid lines (lighter and more subtle)
    for i in range(GRID_CELLS + 1):
        pygame.draw.line(screen, GRAY, (i * CELL_SIZE, 0), (i * CELL_SIZE, GRID_SIZE), 1)
        pygame.draw.line(screen, GRAY, (0, i * CELL_SIZE), (GRID_SIZE, i * CELL_SIZE), 1)

    # Draw pheromones with improved gradient effect
    if draw_pheromones:
        for i in range(GRID_CELLS):
            for j in range(GRID_CELLS):
                if aco.grid[i, j] != 1:  # Don't draw pheromones on obstacles
                    # Normalize pheromone value (between 0 and 1)
                    pheromone_level = min(1.0, aco.pheromones[i, j] / 5.0)

                    if pheromone_level > 0.05:  # Only draw if significant
                        # Create a colorized surface
                        s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)

                        # Calculate a better color gradient based on pheromone strength
                        # Transition from light blue to deep blue
                        r = int(70 - pheromone_level * 40)
                        g = int(130 - pheromone_level * 60)
                        b = int(255)
                        alpha = int(pheromone_level * 180)  # More visible pheromones

                        s.fill((r, g, b, alpha))
                        screen.blit(s, (j * CELL_SIZE, i * CELL_SIZE))

    # Draw best path with glowing effect and gradient
    if aco.best_path:
        # First draw wide glow
        for i in range(len(aco.best_path) - 1):
            current = aco.best_path[i]
            next_pos = aco.best_path[i + 1]

            start_pos = (current[1] * CELL_SIZE + CELL_SIZE // 2,
                         current[0] * CELL_SIZE + CELL_SIZE // 2)
            end_pos = (next_pos[1] * CELL_SIZE + CELL_SIZE // 2,
                       next_pos[0] * CELL_SIZE + CELL_SIZE // 2)

            # Draw glow (wider line with transparency)
            glow_color = (BEST_PATH_COLOR[0], BEST_PATH_COLOR[1], BEST_PATH_COLOR[2], 100)
            glow_surface = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
            pygame.draw.line(glow_surface, glow_color, start_pos, end_pos, 9)
            screen.blit(glow_surface, (0, 0))

            # Draw core path
            pygame.draw.line(screen, BEST_PATH_COLOR, start_pos, end_pos, 4)

    # Draw obstacles, start, and end points with improved visuals
    for i in range(GRID_CELLS):
        for j in range(GRID_CELLS):
            cell_rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            center = cell_rect.center

            if aco.grid[i, j] == 1:  # Obstacle
                # Draw obstacle with shadow effect
                shadow_rect = pygame.Rect(j * CELL_SIZE + 2, i * CELL_SIZE + 2, CELL_SIZE - 2, CELL_SIZE - 2)
                pygame.draw.rect(screen, DARK_GRAY, shadow_rect, border_radius=3)
                pygame.draw.rect(screen, BLACK, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE - 2, CELL_SIZE - 2),
                                 border_radius=3)

                # Add texture to obstacles
                lines = 3
                for line in range(lines):
                    offset = line * (CELL_SIZE // (lines + 1))
                    pygame.draw.line(
                        screen,
                        DARK_GRAY,
                        (j * CELL_SIZE + 3, i * CELL_SIZE + offset + 3),
                        (j * CELL_SIZE + CELL_SIZE - 5, i * CELL_SIZE + offset + 3),
                        1
                    )

            elif aco.grid[i, j] == 2:  # Start
                # Draw rounded rectangle with gradient effect
                pygame.draw.rect(screen, START_COLOR, cell_rect, border_radius=5)

                # Draw "S" marker
                text = FONT_MEDIUM.render("S", True, WHITE)
                text_rect = text.get_rect(center=center)
                # Draw text shadow
                shadow_text = FONT_MEDIUM.render("S", True, DARK_GRAY)
                shadow_rect = shadow_text.get_rect(center=(center[0] + 1, center[1] + 1))
                screen.blit(shadow_text, shadow_rect)
                screen.blit(text, text_rect)

                # Add a highlight
                pygame.draw.circle(screen, WHITE, (center[0] - 3, center[1] - 3), 2)

            elif aco.grid[i, j] == 3:  # End
                # Draw end point with gradient effect
                pygame.draw.rect(screen, END_COLOR, cell_rect, border_radius=5)

                # Draw "E" marker
                text = FONT_MEDIUM.render("E", True, WHITE)
                text_rect = text.get_rect(center=center)
                # Draw text shadow
                shadow_text = FONT_MEDIUM.render("E", True, DARK_GRAY)
                shadow_rect = shadow_text.get_rect(center=(center[0] + 1, center[1] + 1))
                screen.blit(shadow_text, shadow_rect)
                screen.blit(text, text_rect)

                # Add a highlight
                pygame.draw.circle(screen, WHITE, (center[0] - 3, center[1] - 3), 2)

    # Draw ants with better design
    for ant in aco.ants:
        if not ant['done']:
            pos_x = ant['position'][1] * CELL_SIZE + CELL_SIZE // 2
            pos_y = ant['position'][0] * CELL_SIZE + CELL_SIZE // 2

            # Calculate size based on cell size
            ant_size = max(CELL_SIZE // 2.5, 8)

            # Draw ant body (oval)
            pygame.draw.ellipse(
                screen,
                ANT_COLOR,
                (pos_x - ant_size // 2, pos_y - ant_size // 3, ant_size, ant_size * 2 // 3)
            )

            # Draw ant head
            head_size = ant_size // 2
            pygame.draw.circle(
                screen,
                ANT_COLOR,
                (pos_x, pos_y - ant_size // 3),
                head_size // 2
            )

            # Draw legs
            leg_length = ant_size // 2
            for angle in [30, 150, 210, 330]:
                rad = math.radians(angle)
                end_x = pos_x + int(math.cos(rad) * leg_length)
                end_y = pos_y + int(math.sin(rad) * leg_length)
                pygame.draw.line(screen, ANT_COLOR, (pos_x, pos_y), (end_x, end_y), 2)


def draw_sidebar(screen, aco: AntColonyOptimization, buttons: Dict, sliders: Dict,
                 simulation_running: bool, simulation_speed: int, draw_pheromones: bool):
    """Draw the sidebar with controls and information"""
    # Draw sidebar background with gradient effect
    sidebar_rect = pygame.Rect(GRID_SIZE, 0, SIDEBAR_WIDTH, HEIGHT)

    # Create gradient background
    gradient = pygame.Surface((SIDEBAR_WIDTH, HEIGHT))
    for y in range(HEIGHT):
        progress = y / HEIGHT
        color = (
            int(SECONDARY_LIGHT[0] * (1 - progress) + SECONDARY[0] * progress),
            int(SECONDARY_LIGHT[1] * (1 - progress) + SECONDARY[1] * progress),
            int(SECONDARY_LIGHT[2] * (1 - progress) + SECONDARY[2] * progress)
        )
        pygame.draw.line(gradient, color, (0, y), (SIDEBAR_WIDTH, y))

    screen.blit(gradient, (GRID_SIZE, 0))

    # Draw title background panel
    title_rect = pygame.Rect(GRID_SIZE + 10, 10, SIDEBAR_WIDTH - 20, 50)
    pygame.draw.rect(screen, PRIMARY, title_rect, border_radius=10)

    # Add title highlight
    highlight_rect = pygame.Rect(GRID_SIZE + 10, 10, SIDEBAR_WIDTH - 20, 5)
    pygame.draw.rect(screen, PRIMARY_LIGHT, highlight_rect, border_radius=10)

    # Draw title with shadow for better visibility
    title_shadow = FONT_TITLE.render("Ant Colony Pathfinder", True, (30, 30, 30))
    title = FONT_TITLE.render("Ant Colony Pathfinder", True, WHITE)

    screen.blit(title_shadow, (GRID_SIZE + 22, 22))
    screen.blit(title, (GRID_SIZE + 20, 20))

    # Draw section panels with shadows
    # Controls panel
    control_panel = pygame.Rect(GRID_SIZE + 10, 70, SIDEBAR_WIDTH - 20, 190)
    panel_shadow = pygame.Rect(GRID_SIZE + 12, 72, SIDEBAR_WIDTH - 20, 190)
    pygame.draw.rect(screen, DARK_GRAY, panel_shadow, border_radius=10)
    pygame.draw.rect(screen, PANEL_BG, control_panel, border_radius=10)

    # Draw panel header
    section_header = FONT_LARGE.render("Simulation Controls", True, PRIMARY_DARK)
    screen.blit(section_header, (GRID_SIZE + 20, 80))
    # Underline
    pygame.draw.line(screen, PRIMARY, (GRID_SIZE + 20, 105), (GRID_SIZE + SIDEBAR_WIDTH - 30, 105), 2)

    # Parameters panel
    param_panel = pygame.Rect(GRID_SIZE + 10, 270, SIDEBAR_WIDTH - 20, 190)
    param_shadow = pygame.Rect(GRID_SIZE + 12, 272, SIDEBAR_WIDTH - 20, 190)
    pygame.draw.rect(screen, DARK_GRAY, param_shadow, border_radius=10)
    pygame.draw.rect(screen, PANEL_BG, param_panel, border_radius=10)

    # Draw panel header
    param_header = FONT_LARGE.render("Algorithm Parameters", True, PRIMARY_DARK)
    screen.blit(param_header, (GRID_SIZE + 20, 280))
    # Underline
    pygame.draw.line(screen, PRIMARY, (GRID_SIZE + 20, 305), (GRID_SIZE + SIDEBAR_WIDTH - 30, 305), 2)

    # Status panel
    status_panel = pygame.Rect(GRID_SIZE + 10, 470, SIDEBAR_WIDTH - 20, 150)
    status_shadow = pygame.Rect(GRID_SIZE + 12, 472, SIDEBAR_WIDTH - 20, 150)
    pygame.draw.rect(screen, DARK_GRAY, status_shadow, border_radius=10)
    pygame.draw.rect(screen, PANEL_BG, status_panel, border_radius=10)

    # Draw panel header
    status_header = FONT_LARGE.render("Status", True, PRIMARY_DARK)
    screen.blit(status_header, (GRID_SIZE + 20, 480))
    # Underline
    pygame.draw.line(screen, PRIMARY, (GRID_SIZE + 20, 505), (GRID_SIZE + SIDEBAR_WIDTH - 30, 505), 2)

    # Draw buttons
    for button in buttons.values():
        button.draw(screen)

    # Draw sliders
    for slider in sliders.values():
        slider.draw(screen)

    # Draw simulation status with icon
    status_color = ACCENT_SUCCESS if simulation_running else ACCENT_NEUTRAL
    status_icon_rect = pygame.Rect(GRID_SIZE + 25, 520, 12, 12)
    pygame.draw.rect(screen, status_color, status_icon_rect, border_radius=6)

    sim_text = FONT_MEDIUM.render(f"Status: {'Running' if simulation_running else 'Stopped'}", True, BLACK)
    screen.blit(sim_text, (GRID_SIZE + 45, 517))

    # Draw display options with toggle-like appearance
    pheromone_bg = pygame.Rect(GRID_SIZE + 25, 550, 30, 16)
    pygame.draw.rect(screen, GRAY, pheromone_bg, border_radius=8)

    toggle_pos = GRID_SIZE + 25 + (20 if draw_pheromones else 0)
    toggle_rect = pygame.Rect(toggle_pos, 547, 22, 22)
    toggle_color = PRIMARY if draw_pheromones else DARK_GRAY
    pygame.draw.ellipse(screen, toggle_color, toggle_rect)

    # Add highlight to toggle
    if draw_pheromones:
        highlight = pygame.draw.ellipse(screen, PRIMARY_LIGHT, (toggle_pos + 3, 550, 16, 10))

    pheromone_text = FONT_MEDIUM.render("Show Pheromones", True, BLACK)
    screen.blit(pheromone_text, (GRID_SIZE + 60, 547))

    # Draw iteration information with improved styling
    iteration_bg = pygame.Rect(GRID_SIZE + 25, 585, 150, 30)
    pygame.draw.rect(screen, LIGHT_GRAY, iteration_bg, border_radius=5)

    iteration_text = FONT_MEDIUM.render(f"Iterations: {aco.iterations}", True, BLACK)
    screen.blit(iteration_text, (GRID_SIZE + 35, 590))

    # Draw best path information with card styling
    if aco.best_path:
        path_bg = pygame.Rect(GRID_SIZE + 25, 625, 150, 30)
        pygame.draw.rect(screen, LIGHT_GRAY, path_bg, border_radius=5)

        path_text = FONT_MEDIUM.render(f"Best Path: {aco.best_path_length}", True, BLACK)
        screen.blit(path_text, (GRID_SIZE + 35, 630))
    else:
        path_bg = pygame.Rect(GRID_SIZE + 25, 625, 150, 30)
        pygame.draw.rect(screen, LIGHT_GRAY, path_bg, border_radius=5)

        path_text = FONT_MEDIUM.render("No path found yet", True, DARK_GRAY)
        screen.blit(path_text, (GRID_SIZE + 35, 630))

    # Draw convergence information with animated indicator
    if aco.converged:
        # Create pulsing effect
        pulse = (math.sin(pygame.time.get_ticks() / 200) + 1) / 2  # Value between 0 and 1
        pulse_size = int(8 + pulse * 6)  # Size between 8 and 14

        # Draw pulsing indicator
        pygame.draw.circle(screen, ACCENT_SUCCESS, (GRID_SIZE + 35, 675), pulse_size)

        converged_text = FONT_MEDIUM.render("Algorithm Converged!", True, ACCENT_SUCCESS)
        screen.blit(converged_text, (GRID_SIZE + 50, 670))


def draw_info_panel(screen, aco: AntColonyOptimization):
    """Draw the information panel at the bottom"""
    # Draw panel background with shadow
    shadow_rect = pygame.Rect(2, GRID_SIZE + 2, WIDTH - 2, INFO_HEIGHT - 2)
    info_rect = pygame.Rect(0, GRID_SIZE, WIDTH, INFO_HEIGHT)

    pygame.draw.rect(screen, DARK_GRAY, shadow_rect, border_radius=0, border_top_left_radius=0,
                     border_top_right_radius=0)
    pygame.draw.rect(screen, PANEL_BG, info_rect, border_radius=0, border_top_left_radius=0, border_top_right_radius=0)

    # Draw panel divider
    pygame.draw.line(screen, GRAY, (0, GRID_SIZE), (WIDTH, GRID_SIZE), 2)

    # Create chart background
    if aco.iterations > 0:
        chart_rect = pygame.Rect(20, GRID_SIZE + 20, 400, 160)
        chart_shadow = pygame.Rect(22, GRID_SIZE + 22, 400, 160)
        pygame.draw.rect(screen, DARK_GRAY, chart_shadow, border_radius=8)
        pygame.draw.rect(screen, WHITE, chart_rect, border_radius=8)

        # Draw chart title
        chart_title = FONT_MEDIUM.render("Optimization Progress", True, PRIMARY_DARK)
        screen.blit(chart_title, (30, GRID_SIZE + 10))

        # Draw chart
        chart = create_stats_chart(aco)
        screen.blit(chart, (20, GRID_SIZE + 20))

    # Draw instructions in a styled box
    instruction_rect = pygame.Rect(WIDTH - 250, GRID_SIZE + 20, 230, 160)
    instruction_shadow = pygame.Rect(WIDTH - 248, GRID_SIZE + 22, 230, 160)

    pygame.draw.rect(screen, DARK_GRAY, instruction_shadow, border_radius=8)
    pygame.draw.rect(screen, WHITE, instruction_rect, border_radius=8)

    # Instruction header
    inst_header = FONT_MEDIUM.render("Instructions", True, PRIMARY_DARK)
    screen.blit(inst_header, (WIDTH - 240, GRID_SIZE + 25))
    pygame.draw.line(screen, PRIMARY, (WIDTH - 240, GRID_SIZE + 45), (WIDTH - 30, GRID_SIZE + 45), 2)

    # Draw instructions with icons
    instructions = [
        ("🖱️ Left Click", "Add Obstacles"),
        ("🖱️ Right Click", "Remove Obstacles"),
        ("▶️ Start/Stop", "Run simulation"),
        ("⏩ Step", "Single iteration"),
        ("🔄 Reset", "Clear pheromones"),
        ("🗑️ Clear", "Remove obstacles")
    ]

    for i, (icon, text) in enumerate(instructions):
        # Draw icon/action with bold style
        action_text = FONT_SMALL.render(icon, True, PRIMARY_DARK)
        screen.blit(action_text, (WIDTH - 240, GRID_SIZE + 55 + i * 20))

        # Draw description
        desc_text = FONT_SMALL.render(text, True, BLACK)
        screen.blit(desc_text, (WIDTH - 160, GRID_SIZE + 55 + i * 20))

    # Draw a "What is ACO?" explanation box in the middle
    if aco.iterations == 0:  # Only show when simulation hasn't started
        explanation_rect = pygame.Rect(440, GRID_SIZE + 20, 300, 160)
        explanation_shadow = pygame.Rect(442, GRID_SIZE + 22, 300, 160)

        pygame.draw.rect(screen, DARK_GRAY, explanation_shadow, border_radius=8)
        pygame.draw.rect(screen, WHITE, explanation_rect, border_radius=8)

        # Header
        about_header = FONT_MEDIUM.render("What is Ant Colony Optimization?", True, PRIMARY_DARK)
        screen.blit(about_header, (450, GRID_SIZE + 25))
        pygame.draw.line(screen, PRIMARY, (450, GRID_SIZE + 45), (730, GRID_SIZE + 45), 2)

        # Explanation text
        explanation = [
            "Ants initially explore randomly. As they",
            "find paths to the goal, they deposit",
            "pheromone trails. Other ants prefer paths",
            "with stronger pheromone levels.",
            "",
            "Over time, pheromones evaporate from",
            "longer paths, and the colony converges",
            "on the optimal solution."
        ]

        for i, line in enumerate(explanation):
            line_text = FONT_SMALL.render(line, True, BLACK)
            screen.blit(line_text, (450, GRID_SIZE + 55 + i * 16))


def main():
    """Main function to run the ACO visualization"""
    # Initialize screen with icon
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AntColony Pathfinder - AI Visualization")

    # Create an icon for the window (a simple ant shape)
    icon = pygame.Surface((32, 32), pygame.SRCALPHA)
    # Draw ant body
    pygame.draw.ellipse(icon, ANT_COLOR, (8, 12, 16, 10))
    # Draw ant head
    pygame.draw.circle(icon, ANT_COLOR, (24, 16), 5)
    # Draw ant legs
    for angle in [30, 150, 210, 330]:
        rad = math.radians(angle)
        start_x, start_y = 16, 16
        end_x = start_x + int(math.cos(rad) * 10)
        end_y = start_y + int(math.sin(rad) * 10)
        pygame.draw.line(icon, ANT_COLOR, (start_x, start_y), (end_x, end_y), 2)

    pygame.display.set_icon(icon)

    # Initialize clock
    clock = pygame.time.Clock()

    # Initialize ACO
    aco = AntColonyOptimization(GRID_CELLS)

    # Create buttons with better positioning and grouping
    buttons = {
        'start': Button(GRID_SIZE + 30, 120, 130, 40, "Start/Stop"),
        'step': Button(GRID_SIZE + 30, 170, 130, 40, "Step"),
        'reset': Button(GRID_SIZE + 30, 220, 60, 30, "Reset"),
        'clear': Button(GRID_SIZE + 100, 220, 60, 30, "Clear"),
    }

    # Create sliders with improved positioning
    sliders = {
        'ants': Slider(GRID_SIZE + 30, 320, 150, 8, 5, 50, 20, "Number of Ants", "{:.0f}"),
        'alpha': Slider(GRID_SIZE + 30, 360, 150, 8, 0.1, 5.0, 1.0, "Alpha (Pheromone Weight)"),
        'beta': Slider(GRID_SIZE + 30, 400, 150, 8, 0.1, 5.0, 2.0, "Beta (Heuristic Weight)"),
        'evap': Slider(GRID_SIZE + 30, 440, 150, 8, 0.1, 0.9, 0.5, "Evaporation Rate"),
        'speed': Slider(GRID_SIZE + 30, 520, 150, 8, 1, 20, 5, "Simulation Speed", "{:.0f}x")
    }

    # Simulation state
    simulation_running = False
    simulation_speed = 5
    draw_pheromones = True

    # Mouse state
    mouse_down = False
    mouse_button = 0

    # Main loop
    running = True
    last_update = time.time()

    # Additional UI state variables
    hover_cell = None
    show_splash = True
    splash_alpha = 255
    splash_fade_start = time.time() + 2  # Show splash for 2 seconds before fading

    # Create presets for interesting obstacle patterns
    presets = {
        'maze': [
            (10, 10), (10, 11), (10, 12), (10, 13), (10, 15), (10, 16), (10, 17), (10, 18), (10, 19),
            (15, 10), (15, 11), (15, 12), (15, 13), (15, 15), (15, 16), (15, 17), (15, 18), (15, 19),
            (11, 19), (12, 19), (13, 19), (14, 19),
            (20, 10), (20, 11), (20, 12), (20, 13), (20, 15), (20, 16), (20, 17), (20, 18), (20, 19),
            (25, 10), (25, 11), (25, 12), (25, 13), (25, 15), (25, 16), (25, 17), (25, 18), (25, 19),
            (21, 10), (22, 10), (23, 10), (24, 10),
            (30, 10), (30, 11), (30, 12), (30, 13), (30, 15), (30, 16), (30, 17), (30, 18), (30, 19)
        ],
        'spiral': [
            (15, 15), (15, 16), (15, 17), (15, 18), (15, 19), (15, 20), (15, 21), (15, 22), (15, 23), (15, 24),
            (15, 25),
            (16, 25), (17, 25), (18, 25), (19, 25), (20, 25), (21, 25), (22, 25), (23, 25), (24, 25), (25, 25),
            (25, 24), (25, 23), (25, 22), (25, 21), (25, 20), (25, 19), (25, 18), (25, 17), (25, 16), (25, 15),
            (24, 15), (23, 15), (22, 15), (21, 15), (20, 15), (19, 15), (18, 15), (17, 15), (16, 15),
            (16, 16), (16, 17), (16, 18), (16, 19), (16, 20), (16, 21), (16, 22), (16, 23), (16, 24),
            (17, 24), (18, 24), (19, 24), (20, 24), (21, 24), (22, 24), (23, 24), (24, 24),
            (24, 23), (24, 22), (24, 21), (24, 20), (24, 19), (24, 18), (24, 17), (24, 16),
            (23, 16), (22, 16), (21, 16), (20, 16), (19, 16), (18, 16), (17, 16),
        ],
        'blocks': [
            (10, 10), (10, 11), (10, 12), (11, 10), (11, 11), (11, 12), (12, 10), (12, 11), (12, 12),
            (10, 20), (10, 21), (10, 22), (11, 20), (11, 21), (11, 22), (12, 20), (12, 21), (12, 22),
            (20, 10), (20, 11), (20, 12), (21, 10), (21, 11), (21, 12), (22, 10), (22, 11), (22, 12),
            (20, 20), (20, 21), (20, 22), (21, 20), (21, 21), (21, 22), (22, 20), (22, 21), (22, 22),
            (30, 10), (30, 11), (30, 12), (31, 10), (31, 11), (31, 12), (32, 10), (32, 11), (32, 12),
            (30, 20), (30, 21), (30, 22), (31, 20), (31, 21), (31, 22), (32, 20), (32, 21), (32, 22),
        ]
    }

    # Create preset buttons
    preset_buttons = {
        'maze': Button(WIDTH - 240, GRID_SIZE + 150, 60, 25, "Maze", color=ACCENT_INFO),
        'spiral': Button(WIDTH - 170, GRID_SIZE + 150, 60, 25, "Spiral", color=ACCENT_INFO),
        'blocks': Button(WIDTH - 100, GRID_SIZE + 150, 60, 25, "Blocks", color=ACCENT_INFO)
    }

    # Create a toggle button for pheromones
    pheromone_button = Button(GRID_SIZE + 20, 390, 140, 30, "Toggle Pheromones", color=SECONDARY)

    # Create splash screen surface
    splash_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    while running:
        current_time = time.time()
        # Handle events
        current_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Hide splash screen on click
                if show_splash:
                    show_splash = False
                    splash_alpha = 0

                # Check button clicks
                for name, button in buttons.items():
                    if button.check_click(event.pos):
                        if name == 'start':
                            simulation_running = not simulation_running
                        elif name == 'step':
                            aco.run_iteration()
                        elif name == 'reset':
                            aco.reset()
                        elif name == 'clear':
                            aco.clear_obstacles()

                # Check pheromone toggle button
                if pheromone_button.check_click(event.pos):
                    draw_pheromones = not draw_pheromones

                # Check preset buttons
                for name, button in preset_buttons.items():
                    if button.check_click(event.pos):
                        # Clear existing obstacles first
                        aco.clear_obstacles()
                        # Add preset obstacles
                        for obstacle in presets[name]:
                            aco.add_obstacle(obstacle)
                        # Reset algorithm
                        aco.reset()

                # Check slider clicks
                for name, slider in sliders.items():
                    if slider.check_click(event.pos):
                        break

                # Handle grid clicks
                x, y = event.pos
                if x < GRID_SIZE and y < GRID_SIZE:
                    grid_x = y // CELL_SIZE
                    grid_y = x // CELL_SIZE

                    if event.button == 1:  # Left click
                        aco.add_obstacle((grid_x, grid_y))
                        mouse_down = True
                        mouse_button = 1
                    elif event.button == 3:  # Right click
                        aco.remove_obstacle((grid_x, grid_y))
                        mouse_down = True
                        mouse_button = 3

            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_down = False

                # Stop slider dragging
                for slider in sliders.values():
                    slider.stop_drag()

            elif event.type == pygame.MOUSEMOTION:
                # Handle button hovering
                for button in buttons.values():
                    button.check_hover(event.pos)

                # Handle pheromone button hovering
                pheromone_button.check_hover(event.pos)

                # Handle preset button hovering
                for button in preset_buttons.values():
                    button.check_hover(event.pos)

                # Handle slider hovering and dragging
                for name, slider in sliders.items():
                    slider.check_hover(event.pos)
                    slider.update(event.pos)
                    if name == 'speed':
                        simulation_speed = int(slider.value)

                # Handle grid hovering and drawing
                if event.pos[0] < GRID_SIZE and event.pos[1] < GRID_SIZE:
                    grid_x = event.pos[1] // CELL_SIZE
                    grid_y = event.pos[0] // CELL_SIZE
                    hover_cell = (grid_x, grid_y)

                    if mouse_down:
                        if mouse_button == 1:  # Left button (add obstacles)
                            aco.add_obstacle((grid_x, grid_y))
                        elif mouse_button == 3:  # Right button (remove obstacles)
                            aco.remove_obstacle((grid_x, grid_y))
                else:
                    hover_cell = None

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    simulation_running = not simulation_running
                elif event.key == pygame.K_p:
                    draw_pheromones = not draw_pheromones
                elif event.key == pygame.K_r:
                    aco.reset()
                elif event.key == pygame.K_c:
                    aco.clear_obstacles()
                elif event.key == pygame.K_s:
                    aco.run_iteration()
                elif event.key == pygame.K_ESCAPE:
                    running = False

        # Update ACO parameters from sliders
        aco.set_parameters(
            int(sliders['ants'].value),
            sliders['alpha'].value,
            sliders['beta'].value,
            sliders['evap'].value
        )

        # Run simulation based on speed
        if simulation_running:
            if current_time - last_update > 1.0 / simulation_speed:
                aco.run_iteration()
                last_update = current_time

        # Draw everything
        screen.fill(BG_COLOR)
        draw_grid(screen, aco, draw_pheromones)

        # Draw hover highlight on grid
        if hover_cell and hover_cell[0] < GRID_CELLS and hover_cell[1] < GRID_CELLS:
            hover_rect = pygame.Rect(
                hover_cell[1] * CELL_SIZE,
                hover_cell[0] * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )
            # Only show hover if not on start or end points
            if (aco.grid[hover_cell] != 2 and aco.grid[hover_cell] != 3):
                s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                s.fill((255, 255, 255, 50))  # Semi-transparent white
                screen.blit(s, (hover_cell[1] * CELL_SIZE, hover_cell[0] * CELL_SIZE))
                pygame.draw.rect(screen, PRIMARY, hover_rect, 2)

        # Draw UI elements
        draw_sidebar(screen, aco, buttons, sliders, simulation_running, simulation_speed, draw_pheromones)
        pheromone_button.draw(screen)
        draw_info_panel(screen, aco)

        # Draw preset buttons
        for button in preset_buttons.values():
            button.draw(screen)

        # Draw splash screen with fade effect
        if show_splash:
            if current_time > splash_fade_start:
                # Fade out
                splash_alpha = max(0, splash_alpha - 5)
                if splash_alpha == 0:
                    show_splash = False

            # Create splash content
            splash_surface.fill((0, 0, 0, 0))  # Clear with transparent

            # Add semi-transparent background
            s = pygame.Surface((WIDTH, HEIGHT))
            s.fill(BG_COLOR)
            splash_surface.blit(s, (0, 0))

            # Add title
            splash_title = FONT_TITLE.render("AntColony Pathfinder", True, PRIMARY)
            title_rect = splash_title.get_rect(center=(WIDTH // 2, HEIGHT // 3))
            splash_surface.blit(splash_title, title_rect)

            # Add subtitle
            splash_subtitle = FONT_LARGE.render("Swarm Intelligence Visualization", True, PRIMARY_DARK)
            subtitle_rect = splash_subtitle.get_rect(center=(WIDTH // 2, HEIGHT // 3 + 50))
            splash_surface.blit(splash_subtitle, subtitle_rect)

            # Add information
            info_lines = [
                "Watch as virtual ants find optimal paths through a maze of obstacles.",
                "Ants use pheromones to communicate and collectively solve pathfinding problems.",
                "",
                "Click anywhere to begin exploring...",
            ]

            for i, line in enumerate(info_lines):
                info_text = FONT_MEDIUM.render(line, True, BLACK)
                info_rect = info_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 30 + i * 30))
                splash_surface.blit(info_text, info_rect)

            # Set alpha for the entire surface
            splash_surface.set_alpha(splash_alpha)
            screen.blit(splash_surface, (0, 0))

        # Update display
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()