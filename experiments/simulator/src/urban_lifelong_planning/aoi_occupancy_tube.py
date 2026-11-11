"""Deterministic AoI-expanded future occupancy tubes for Stage 4."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, hypot

import numpy as np

Point = tuple[int, int]


@dataclass(frozen=True)
class AOITubeConfig:
    base_buffer: int = 0
    aoi_scale: float = 0.5
    max_buffer: int = 2

    def buffer(self, aoi: int) -> int:
        return min(self.max_buffer, self.base_buffer + ceil(self.aoi_scale * max(0, aoi)))


def occupancy_tube(buildings: np.ndarray, nominal_path: list[Point], buffer: int) -> list[set[Point]]:
    """Expand each nominal free-space position by an Euclidean grid buffer."""
    height, width = buildings.shape
    tube: list[set[Point]] = []
    for x, y in nominal_path:
        occupied: set[Point] = set()
        for dy in range(-buffer, buffer + 1):
            for dx in range(-buffer, buffer + 1):
                nx, ny = x + dx, y + dy
                if hypot(dx, dy) <= buffer + 1e-9 and 0 <= nx < width and 0 <= ny < height and not buildings[ny, nx]:
                    occupied.add((nx, ny))
        tube.append(occupied)
    return tube


def tube_conflict_count(tubes: list[list[set[Point]]]) -> int:
    """Diagnostic overlap count; Stage 5 turns these overlaps into a graph."""
    conflicts = 0
    for left in range(len(tubes)):
        for right in range(left + 1, len(tubes)):
            if any(a & b for a, b in zip(tubes[left], tubes[right])):
                conflicts += 1
    return conflicts
