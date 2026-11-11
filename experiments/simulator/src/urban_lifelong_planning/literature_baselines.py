"""Source-based BaF geometric guidance in the common RHCR reservation solver.

This is explicitly an adaptation, not a reproduction of tBaF, ComPaths, or
cBaF. Algorithm source: Aslan (2024), DOI 10.1007/s00607-024-01309-7,
Section 4, Eqs. (5)-(6). The requested IISEC 2026 article
10.1109/IISEC69317.2026.11418413 studies this BaF heuristic.

Retained: perpendicular equally spaced sample lines; nearest safe sample
selection in both directions; most-similar matched segment; four candidates
with sequential greedy midpoint updates. Adapted: finite map bounds, grid
obstacles and no-corner-cutting clearance, and use as guidance for the shared
reservation solver. No privileged execution state is read here.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import acos, ceil, floor, hypot, pi
import numpy as np

from .motion_graph import legal_cell, legal_transition
from .rhcr_adapted import (RHCRAdapted, RHCRConfig, ReservationTable,
                          SpaceTimePrioritizedPlanner, validate_joint_paths)

Point = tuple[int, int]
BASELINE_METHODS = ("BaF-guided RHCR",)


@dataclass(frozen=True)
class BaFConfig:
    # Original study considers D=10,15,20,25. Fixed before outcome inspection.
    points: int = 15
    turn_weight: float = .05
    maximum_turn: float = pi / 4

    def __post_init__(self):
        if self.points < 2 or self.turn_weight < 0:
            raise ValueError("require points >= 2 and turn_weight >= 0")


def _cell(point) -> Point:
    return floor(point[0] + .5), floor(point[1] + .5)


def _raster_segment(a, b) -> list[Point]:
    """Dense rounding preserves every grid crossing, including corner tests."""
    ax, ay, bx, by = float(a[0]), float(a[1]), float(b[0]), float(b[1])
    count = max(1, ceil(max(abs(bx-ax),abs(by-ay))*5))
    result = []
    for i in range(count+1):
        point = floor(ax+(bx-ax)*i/count+.5), floor(ay+(by-ay)*i/count+.5)
        if not result or point != result[-1]:
            result.append(point)
    return result


def _score(path: np.ndarray, buildings: np.ndarray, cfg: BaFConfig):
    """Hard clearance first, then length and excessive turning cost.

    A finite lexicographic violation count enables midpoint candidate repair;
    only candidates with zero violations may become an operational guide.
    """
    violations = 0
    vectors = np.diff(path, axis=0)
    length = float(np.linalg.norm(vectors, axis=1).sum())
    for a, b in zip(path, path[1:]):
        cells = _raster_segment(a, b)
        violations += sum(not legal_transition(buildings, p, q)
                          for p, q in zip(cells, cells[1:]))
        violations += int(not legal_cell(buildings, cells[0]))
    turn = 0.0
    for a, b in zip(vectors, vectors[1:]):
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom:
            angle = acos(float(np.clip(np.dot(a, b) / denom, -1., 1.)))
            if angle > cfg.maximum_turn:
                turn += angle
    return violations, length + cfg.turn_weight * turn


def baf_candidates(buildings: np.ndarray, start: Point, goal: Point,
                   cfg: BaFConfig = BaFConfig()) -> list[np.ndarray]:
    """Return the two greedy parents and four candidates of BaF Section 4.

    Empty output means at least one perpendicular layer has no safe sample.
    Candidate segments still require clearance validation by the caller.
    """
    if not legal_cell(buildings, start) or not legal_cell(buildings, goal):
        return []
    origin, target = np.array(start, float), np.array(goal, float)
    distance = float(np.linalg.norm(target-origin))
    if not distance:
        return [np.array([origin, target])]
    along = (target-origin) / distance
    across = np.array([-along[1], along[0]])
    height, width = buildings.shape
    corners = np.array([[0,0], [width-1,0], [0,height-1], [width-1,height-1]], float)
    low, high = (corners-origin) @ across, (corners-origin) @ across
    offsets = np.linspace(float(low.min()), float(high.max()), cfg.points+2)[1:-1]
    layers = [origin + along*(distance*i/(cfg.points+1)) + offsets[:,None]*across
              for i in range(1,cfg.points+1)]

    def greedy(begin, ordered_layers, end):
        points = [begin]
        for layer in ordered_layers:
            valid = [p for p in layer if legal_cell(buildings, _cell(p))]
            if not valid:
                return None
            # Nearest point; stable sample order resolves exact distance ties.
            points.append(min(valid, key=lambda p: float(np.linalg.norm(p-points[-1]))))
        return np.array(points+[end])

    forward = greedy(origin, layers, target)
    reverse = greedy(target, layers[::-1], origin)
    if forward is None or reverse is None:
        return []
    backward = reverse[::-1].copy()
    parents = [forward, backward]
    score_cache = {}
    def cost(path):
        key = path.tobytes()
        if key not in score_cache:
            score_cache[key] = _score(path, buildings, cfg)
        return score_cache[key]
    scores = [cost(p) for p in parents]
    # Eq. (5): maximizing reciprocal sum is minimizing its denominator.
    differences = np.linalg.norm(forward-backward, axis=1)
    j = 1 + int(np.argmin(differences[1:-2] + differences[2:-1]))
    winner = parents[0 if scores[0] <= scores[1] else 1]
    candidates = list(parents)
    for parent in parents:
        for search_indices in (range(j+2, cfg.points+1), range(1,j)):
            candidate = parent.copy()
            candidate[j:j+2] = winner[j:j+2]
            best_score = cost(candidate)
            for i in search_indices:
                proposal = candidate.copy()
                proposal[i] = (forward[i]+backward[i]) / 2  # Eq. (6)
                score = cost(proposal)
                if score < best_score:
                    candidate, best_score = proposal, score
            candidates.append(candidate)
    return candidates


def baf_route(buildings: np.ndarray, start: Point, goal: Point,
              cfg: BaFConfig = BaFConfig()) -> np.ndarray | None:
    candidates = baf_candidates(buildings, start, goal, cfg)
    if not candidates:
        return None
    route = min(candidates, key=lambda p: _score(p, buildings, cfg))
    return route if _score(route, buildings, cfg)[0] == 0 else None


def _lookahead(route: np.ndarray, length: float) -> tuple[float,float]:
    for start, end in zip(route, route[1:]):
        segment = float(np.linalg.norm(end-start))
        if segment >= length and segment:
            point = start + (end-start)*length/segment
            return float(point[0]), float(point[1])
        length -= segment
    return tuple(float(v) for v in route[-1])


class _GuidedSearch(SpaceTimePrioritizedPlanner):
    """Shared temporal search with an explicitly scheduled BaF subgoal.

    An intermediate guide may need fewer than ``window`` graph steps even if
    its Euclidean arc length equals ``window``. Treating it only as a terminal
    ranking target lets wait-first BFS prepend slack and can freeze every
    executed prefix. The shared earliest-arrival termination instead pads
    safe waits AFTER reaching the guide. Environment task completion still
    uses the actual assigned target, never this internal subgoal.
    """
    guide: tuple[float,float] | None = None

    def _goal_reached(self, point, goal):
        return (super()._goal_reached(point, goal) or
                (self.guide is not None and
                 hypot(point[0]-self.guide[0],point[1]-self.guide[1]) <= self.goal_radius))

    def _goal_key(self, point, goal):
        if self.guide is None:
            return super()._goal_key(point, goal)
        return (hypot(point[0]-self.guide[0], point[1]-self.guide[1]),
                hypot(point[0]-goal[0], point[1]-goal[1]), point[1], point[0])


class BaFGuidedRHCR(RHCRAdapted):
    def __init__(self, config: RHCRConfig = RHCRConfig(), baf: BaFConfig = BaFConfig()):
        super().__init__(config)
        self.baf = baf
        self.last_route_generated_count = 0
        self.last_route_fallback_count = 0
        self._route_cache = {}
        self._map_object = None

    def plan(self, buildings, positions, goals):
        if len(positions) != len(goals):
            raise ValueError("positions and goals must have identical length")
        self._check_starts(buildings, positions)
        self.last_plan_fallback = False
        self.last_failure_reason = ""
        self.last_route_generated_count = self.last_route_fallback_count = 0
        if self._map_object is not buildings:
            self._route_cache.clear()
            self._map_object = buildings
        reservations, paths = ReservationTable(), []
        search = _GuidedSearch(buildings, goal_radius=1.)
        for owner, (position, goal) in enumerate(zip(positions, goals)):
            key = (position, goal)
            if key not in self._route_cache:
                self._route_cache[key] = baf_route(buildings, position, goal, self.baf)
            route = self._route_cache[key]
            if route is None:
                self.last_route_fallback_count += 1
                search.guide = None
            else:
                self.last_route_generated_count += 1
                search.guide = _lookahead(route, float(self.config.planning_window))
            reservations.vertices[0] = {
                p: {i} for i,p in enumerate(positions) if i != owner}
            path = search.path(position, goal, self.config.planning_window, reservations)
            if path is None:
                return self._wait_fallback(buildings, positions, "BaF-guided prioritized search failed")
            reservations.reserve(path, owner)
            paths.append(path)
        if not validate_joint_paths(buildings, paths, positions, self.config.planning_window):
            return self._wait_fallback(buildings, positions, "BaF-guided joint validation failed")
        return paths


def make_planner(method: str, config: RHCRConfig) -> RHCRAdapted:
    if method == BASELINE_METHODS[0]:
        return BaFGuidedRHCR(config)
    raise ValueError(f"unknown source-based planner: {method}")
