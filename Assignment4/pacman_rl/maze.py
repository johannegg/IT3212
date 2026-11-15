"""Reading the MsPacman maze from the screen and building a navigation graph.

The Atari 2600 MsPacman maze lies on a regular grid:
  * rows are 12 pixels tall (row r is centered at y = 7.5 + 12 r),
  * columns are 4 pixels wide (column c covers x = 4c .. 4c+3).
Walls and dots share the same pink color. Dots are tiny blobs (8 pixels),
power pellets are small blobs (~28 pixels, and they blink), walls are large.

A node (r, c) is a place where Ms. Pac-Man's center can be. A node is valid if
the columns c-1, c and c+1 in row r are free of walls (her sprite is 7 pixels
wide). Horizontal edges have length 1 (4 px) and vertical edges length 3
(12 px), so graph distances are measured in units of 4 pixels.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path

PINK = np.array([228, 111, 111], dtype=np.uint8)
BACKGROUND = np.array([0, 28, 136], dtype=np.uint8)
MAZE_HEIGHT = 172           # pixel rows of the play area
N_ROWS, N_COLS = 14, 40
ROW_H, COL_W = 12, 4
VERTICAL_COST = 3           # 12 px / 4 px

# RAM positions -> pixel centers (calibrated against the screen, see README)
PAC_X_OFF, PAC_Y_OFF = -8.2, 5.4
GHOST_X_OFF, GHOST_Y_OFF = -8.4, 5.9


def pixel_to_cell(px: float, py: float) -> tuple[int, int]:
    r = int(round((py - 7.5) / ROW_H))
    c = int(round((px - 2.0) / COL_W))
    return min(max(r, 0), N_ROWS - 1), c % N_COLS


class Maze:
    """Walls, nodes and all-pairs distances for one maze layout."""

    def __init__(self, frame: np.ndarray):
        img = frame[:MAZE_HEIGHT]
        pink = (img == PINK).all(-1)
        lab, n = ndimage.label(pink)
        sizes = ndimage.sum(pink, lab, range(1, n + 1))
        big = np.isin(lab, 1 + np.where(sizes > 60)[0])
        self.walls = big

        # free[r, c]: no wall pixel inside cell (r, c)
        free = np.ones((N_ROWS, N_COLS), dtype=bool)
        for r in range(N_ROWS):
            y0, y1 = ROW_H * r + 2, min(ROW_H * r + 14, MAZE_HEIGHT)
            for c in range(N_COLS):
                free[r, c] = not big[y0:y1, COL_W * c:COL_W * c + 4].any()
        self.free = free

        valid = np.zeros_like(free)
        for r in range(N_ROWS):
            for c in range(N_COLS):
                valid[r, c] = free[r, (c - 1) % N_COLS] and free[r, c] and free[r, (c + 1) % N_COLS]
        self.valid = valid

        self.node_id = -np.ones((N_ROWS, N_COLS), dtype=int)
        coords = np.argwhere(valid)
        for i, (r, c) in enumerate(coords):
            self.node_id[r, c] = i
        self.coords = coords
        n_nodes = len(coords)

        # neighbor[i, d] for d in (UP, RIGHT, LEFT, DOWN); -1 if blocked
        self.neighbor = -np.ones((n_nodes, 4), dtype=int)
        adj = lil_matrix((n_nodes, n_nodes))
        for i, (r, c) in enumerate(coords):
            for d, (dr, dc, w) in enumerate([(-1, 0, VERTICAL_COST), (0, 1, 1), (0, -1, 1), (1, 0, VERTICAL_COST)]):
                rr, cc = r + dr, (c + dc) % N_COLS
                if 0 <= rr < N_ROWS and valid[rr, cc]:
                    j = self.node_id[rr, cc]
                    self.neighbor[i, d] = j
                    adj[i, j] = w
        self.dist = shortest_path(adj.tocsr(), method="D", directed=False)

        # nearest valid node for every grid cell (used to snap positions)
        self.snap = np.zeros((N_ROWS, N_COLS), dtype=int)
        for r in range(N_ROWS):
            for c in range(N_COLS):
                dc = np.minimum(np.abs(coords[:, 1] - c), N_COLS - np.abs(coords[:, 1] - c))
                cost = dc + VERTICAL_COST * np.abs(coords[:, 0] - r)
                self.snap[r, c] = int(np.argmin(cost))

        # dot positions (small blobs) and their node ids
        dot_labels = 1 + np.where(sizes <= 12)[0]
        centers = ndimage.center_of_mass(pink, lab, dot_labels)
        self.dot_pixels = np.array([[int(np.floor(y)), int(np.floor(x))] for y, x in centers], dtype=int)
        self.dot_nodes = np.array([self.node_of_pixel(x, y) for y, x in centers], dtype=int)

    def node_of_pixel(self, px: float, py: float) -> int:
        r, c = pixel_to_cell(px, py)
        return int(self.snap[r, c])

    def find_power_pellets(self, frame: np.ndarray) -> list[tuple[int, int]]:
        """Power pellets visible in this frame (they blink), as pixel centers."""
        img = frame[:MAZE_HEIGHT]
        pink = (img == PINK).all(-1) & ~self.walls
        lab, n = ndimage.label(pink)
        if n == 0:
            return []
        sizes = ndimage.sum(pink, lab, range(1, n + 1))
        keep = 1 + np.where((sizes > 12) & (sizes <= 60))[0]
        return [(float(x), float(y)) for y, x in ndimage.center_of_mass(pink, lab, keep)]

    def dots_present(self, frame: np.ndarray) -> np.ndarray:
        """A dot is present unless its pixel shows the background color.
        (A sprite drawn on top of a dot does not count as eating it.)"""
        px = frame[self.dot_pixels[:, 0], self.dot_pixels[:, 1]]
        return ~(px == BACKGROUND).all(-1)
