"""Common free-cell, eight-neighbour motion graph with no corner cutting."""
from __future__ import annotations

import numpy as np

Point = tuple[int, int]


def legal_cell(buildings: np.ndarray, point: Point) -> bool:
    x, y = point
    height, width = buildings.shape
    return 0 <= x < width and 0 <= y < height and not buildings[y, x]


def legal_transition(buildings: np.ndarray, start: Point, end: Point) -> bool:
    """Test a graph edge or hold, including both endpoints and diagonal clearance."""
    if not legal_cell(buildings, start) or not legal_cell(buildings, end):
        return False
    dx, dy = end[0] - start[0], end[1] - start[1]
    if max(abs(dx), abs(dy)) > 1:
        return False
    return not (dx and dy) or (not buildings[start[1], end[0]] and not buildings[end[1], start[0]])
