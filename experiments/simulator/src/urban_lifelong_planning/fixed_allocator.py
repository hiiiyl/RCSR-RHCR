"""Shared fixed task allocator; it is intentionally not a method contribution."""
from __future__ import annotations

import math
from collections import deque

import numpy as np
from .motion_graph import legal_transition

Point = tuple[int, int]


def reference_eight_neighbor_distance_map(buildings: np.ndarray, target: Point) -> np.ndarray:
    """Obstacle-aware unit-step distance map for the public task allocator."""
    height, width = buildings.shape
    tx, ty = target
    distances = np.full((height, width), np.inf, dtype=np.float64)
    if not (0 <= tx < width and 0 <= ty < height) or buildings[ty, tx]:
        return distances
    distances[ty, tx] = 0.0
    queue: deque[Point] = deque([(tx, ty)])
    moves = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
    while queue:
        x, y = queue.popleft()
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if legal_transition(buildings, (x, y), (nx, ny)) and not math.isfinite(distances[ny, nx]):
                distances[ny, nx] = distances[y, x] + 1.0
                queue.append((nx, ny))
    return distances


def eight_neighbor_distance_map(buildings: np.ndarray, target: Point) -> np.ndarray:
    """Content-keyed exact distances, identical to the reference unit-step BFS.

    Public static map and target only. Immutable cached arrays prevent callers
    from corrupting later assignments. Radius zero preserves V1 cost semantics.
    """
    from .rhcr_adapted import _goal_distance_map
    return _goal_distance_map(buildings.shape, np.asarray(buildings, dtype=np.bool_).tobytes(), target, 0.0)


def hungarian_min_cost(cost: np.ndarray) -> list[tuple[int, int]]:
    """Return row/column assignments using the O(n^3) Hungarian algorithm."""
    matrix = np.asarray(cost, dtype=np.float64)
    rows, cols = matrix.shape
    if rows == 0 or cols == 0:
        return []
    if rows > cols:
        return [(column, row) for row, column in hungarian_min_cost(matrix.T)]
    u = np.zeros(rows + 1)
    v = np.zeros(cols + 1)
    p = np.zeros(cols + 1, dtype=int)
    way = np.zeros(cols + 1, dtype=int)
    for row in range(1, rows + 1):
        p[0] = row
        column0 = 0
        min_values = np.full(cols + 1, np.inf)
        used = np.zeros(cols + 1, dtype=bool)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = np.inf
            column1 = 0
            for column in range(1, cols + 1):
                if not used[column]:
                    current = matrix[row0 - 1, column - 1] - u[row0] - v[column]
                    if current < min_values[column]:
                        min_values[column] = current
                        way[column] = column0
                    if min_values[column] < delta:
                        delta = min_values[column]
                        column1 = column
            for column in range(cols + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    min_values[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    return [(p[column] - 1, column - 1) for column in range(1, cols + 1) if p[column] != 0]


class FixedAllocator:
    """Assign idle UAVs to pending targets by one deterministic global matching."""

    def __init__(self, age_weight: float = 0.0):
        if not math.isfinite(age_weight) or age_weight < 0:
            raise ValueError("age_weight must be finite and nonnegative")
        self.age_weight = float(age_weight)

    def allocate(self, buildings: np.ndarray, positions: list[Point], idle_uavs: list[int], target_positions: dict[int, Point], target_ages: dict[int, int] | None = None) -> list[tuple[int, int]]:
        target_ids = sorted(target_positions)
        if not idle_uavs or not target_ids:
            return []
        distances = [eight_neighbor_distance_map(buildings, target_positions[target_id]) for target_id in target_ids]
        cost = np.empty((len(idle_uavs), len(target_ids)), dtype=np.float64)
        for row, drone_idx in enumerate(idle_uavs):
            x, y = positions[drone_idx]
            for column, distance_map in enumerate(distances):
                value = float(distance_map[y, x])
                # Public age credit trades one step of travel for 1/weight
                # waiting steps. Existing assignments remain sticky. Unreachable
                # tasks receive no credit, preserving the V1 finite penalty.
                age = max(0, (target_ages or {}).get(target_ids[column], 0))
                cost[row, column] = value - self.age_weight * age if math.isfinite(value) else 1e6
        return [(idle_uavs[row], target_ids[column]) for row, column in hungarian_min_cost(cost)]
