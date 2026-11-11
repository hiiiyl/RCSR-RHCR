"""Stage-2 RHCR-adapted: global rolling-horizon prioritized planning only."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from math import hypot

import numpy as np

from .aoi_occupancy_tube import occupancy_tube
from .delayed_state_channel import StateMessage
from .motion_graph import legal_cell, legal_transition

Point = tuple[int, int]
TRACKING_MOVES = ((0, 0), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))


def reference_projection(buildings: np.ndarray, position: Point, reference: Point) -> Point:
    """The deterministic reference tracker used by execution and EMC belief."""
    candidates = [
        (position[0] + dx, position[1] + dy)
        for dx, dy in TRACKING_MOVES
        if legal_transition(buildings, position, (position[0] + dx, position[1] + dy))
    ]
    if not candidates:
        raise RuntimeError("a valid UAV position must have a legal hold successor")
    return min(candidates, key=lambda point: (hypot(point[0] - reference[0], point[1] - reference[1]), point[1], point[0]))


@dataclass(frozen=True)
class RHCRConfig:
    planning_window: int = 8
    execution_horizon: int = 2

    def __post_init__(self):
        if self.planning_window < 1 or not 1 <= self.execution_horizon <= self.planning_window:
            raise ValueError("require 1 <= execution_horizon <= planning_window")


@dataclass(frozen=True)
class IssuedPathAction:
    """A controller-issued command, retained independently of execution."""

    timestamp: int
    path_id: int
    path_index: int
    explicit_hover: bool


@dataclass(frozen=True)
class ProgressStartEstimate:
    """MAP start selected from delayed, confirmed path-progress evidence."""

    path_id: int
    map_index: int
    position: Point
    distribution: tuple[tuple[Point, float], ...]
    evaluated_cross_plan_transition_count: int
    cross_plan_unreachable_transition_count: int
    evaluated_cross_plan_probability_mass: float
    rejected_probability_mass: float


@dataclass(frozen=True)
class EMCStartEstimate:
    """World-coordinate belief and diagnostics for execution-model consistency."""

    position: Point
    distribution: tuple[tuple[Point, float], ...]
    reference_projection_rate: float
    mean_reference_gap: float
    mean_remaining_gap_after_projection: float


@dataclass(frozen=True)
class EMCTransition:
    """Audit-visible result of one reference-tracker belief update."""

    belief: tuple[tuple[Point, float], ...]
    reference_projection_rate: float
    mean_reference_gap: float
    mean_remaining_gap_after_projection: float


class ExecutionProgressStartEstimator:
    """Fixed-probability path-index propagation for EP-RHCR.

    The estimator deliberately has no environment state input.  It resets to
    each newly delivered status message, then replays controller-issued
    actions after that timestamp.  Hypotheses carry both a spatial position
    and plan-local index. A non-hover command can advance with probability
    ``1 - deviation_probability`` only when its waypoint is a legal one-step
    successor of that hypothesis position. An explicit hover never advances
    an index.
    """

    physical_anchor_belief = True

    def __init__(self, deviation_probability: float = 0.25, buildings: np.ndarray | None = None):
        if not 0.0 <= deviation_probability <= 1.0:
            raise ValueError("deviation_probability must lie in [0, 1]")
        self.deviation_probability = float(deviation_probability)
        # A missing mask retains the historical obstacle-free unit-test API.
        # All experiments must pass the actual map to enforce the graph.
        self.buildings = buildings

    def estimate(self, message: StateMessage, paths_by_id: dict[int, list[Point]], actions: list[IssuedPathAction], current_step: int) -> ProgressStartEstimate:
        path = paths_by_id.get(message.path_id)
        if self.buildings is not None and not legal_cell(self.buildings, message.position):
            raise ValueError("delivered position is outside the legal graph")
        # Launch packets have no path ID but still anchor subsequent commands.
        start_index = 0 if path is None else min(max(message.path_index, 0), len(path) - 1)
        distribution: dict[tuple[Point, int, int], float] = {(message.position, message.path_id, start_index): 1.0}
        has_progress_command = False
        evaluated_cross_plan_transitions = 0
        unreachable_transitions = 0
        evaluated_cross_plan_mass = 0.0
        rejected_probability_mass = 0.0
        for action in actions:
            action_path = paths_by_id.get(action.path_id)
            if action.timestamp <= message.timestamp or action.timestamp > current_step or action_path is None:
                continue
            if action.explicit_hover:
                continue
            has_progress_command = True
            advanced: dict[tuple[Point, int, int], float] = {}
            target = action_path[action.path_index]
            for (position, state_path_id, index), probability in distribution.items():
                dx, dy = target[0] - position[0], target[1] - position[1]
                reachable = max(abs(dx), abs(dy)) == 1 and (
                    self.buildings is None or legal_transition(self.buildings, position, target)
                )
                cross_plan = action.path_id != state_path_id
                if cross_plan:
                    evaluated_cross_plan_transitions += 1
                    evaluated_cross_plan_mass += probability
                if reachable:
                    if self.deviation_probability:
                        advanced[(position, state_path_id, index)] = advanced.get((position, state_path_id, index), 0.0) + probability * self.deviation_probability
                    if self.deviation_probability < 1.0:
                        advanced[(target, action.path_id, action.path_index)] = advanced.get((target, action.path_id, action.path_index), 0.0) + probability * (1.0 - self.deviation_probability)
                else:
                    advanced[(position, state_path_id, index)] = advanced.get((position, state_path_id, index), 0.0) + probability
                    if cross_plan:
                        unreachable_transitions += 1
                        rejected_probability_mass += probability
            distribution = advanced
        if not has_progress_command:
            # A newly delivered state packet is the authoritative historical
            # correction.  With no later move command to propagate, use that
            # packet position directly (not a nominal path cell).
            return ProgressStartEstimate(message.path_id, start_index, message.position, ((message.position, 1.0),), 0, 0, 0.0, 0.0)
        position_mass: dict[Point, float] = {}
        for (position, _, _), probability in distribution.items():
            position_mass[position] = position_mass.get(position, 0.0) + probability
        # Aggregate by physical position before selecting the MAP planning
        # start. Ties are resolved deterministically without environment state.
        map_position = max(position_mass, key=lambda point: (position_mass[point], point[1], point[0]))
        representatives = [state for state in distribution if state[0] == map_position]
        _, map_path_id, map_index = max(representatives, key=lambda state: (distribution[state], state[2], state[1]))
        return ProgressStartEstimate(map_path_id, map_index, map_position, tuple(sorted(position_mass.items())), evaluated_cross_plan_transitions, unreachable_transitions, evaluated_cross_plan_mass, rejected_probability_mass)


class IndexOnlyProgressStartEstimator:
    """Frozen index-only EP ablation estimator.

    This ablation deliberately represents progress solely by ``(plan_id,
    path_index)``.  A non-hover issued action transfers ``1-p`` mass to its
    plan-local index and leaves ``p`` mass at the previous index.  It neither
    evaluates spatial reachability nor aggregates posterior mass by world
    position.  A delivered message re-anchors the plan-local state before the
    fixed post-message action-history replay.
    """

    physical_anchor_belief = False

    def __init__(self, deviation_probability: float = 0.25):
        if not 0.0 <= deviation_probability <= 1.0:
            raise ValueError("deviation_probability must lie in [0, 1]")
        self.deviation_probability = float(deviation_probability)

    def estimate(
        self,
        message: StateMessage,
        paths_by_id: dict[int, list[Point]],
        actions: list[IssuedPathAction],
        current_step: int,
    ) -> ProgressStartEstimate:
        message_path = paths_by_id.get(message.path_id)
        if message_path is None:
            # Use a synthetic local anchor so launch packets can replay commands.
            paths_by_id = dict(paths_by_id)
            paths_by_id[message.path_id] = [message.position]
            message_path = paths_by_id[message.path_id]
        start_index = min(max(message.path_index, 0), len(message_path) - 1)
        distribution: dict[tuple[int, int], float] = {(message.path_id, start_index): 1.0}
        for action in actions:
            action_path = paths_by_id.get(action.path_id)
            if action.timestamp <= message.timestamp or action.timestamp > current_step or action_path is None:
                continue
            if action.explicit_hover:
                continue
            updated: dict[tuple[int, int], float] = {}
            for state, probability in distribution.items():
                updated[state] = updated.get(state, 0.0) + probability * self.deviation_probability
                advanced = (action.path_id, min(max(action.path_index, 0), len(action_path) - 1))
                updated[advanced] = updated.get(advanced, 0.0) + probability * (1.0 - self.deviation_probability)
            distribution = updated
        map_path_id, map_index = max(distribution, key=lambda state: (distribution[state], state[1], state[0]))
        map_path = paths_by_id.get(map_path_id)
        position = message.position if map_path is None else map_path[min(map_index, len(map_path) - 1)]
        # The public distribution field remains spatial for common diagnostics,
        # but its MAP decision above was intentionally made in plan-index space.
        position_distribution = tuple(sorted((paths_by_id[path_id][index], probability) for (path_id, index), probability in distribution.items()))
        return ProgressStartEstimate(map_path_id, map_index, position, position_distribution, 0, 0, 0.0, 0.0)


class ExecutionModelConsistentEstimator:
    """Reconstruct delayed execution belief with the deployed reference tracker.

    Only world-coordinate positions are belief states. Plan identifiers and
    path indices identify historical reference waypoints; they are never part
    of the stochastic state.
    """

    physical_anchor_belief = True

    def __init__(self, buildings: np.ndarray, deviation_probability: float = 0.25):
        if not 0.0 <= deviation_probability <= 1.0:
            raise ValueError("deviation_probability must lie in [0, 1]")
        self.buildings = buildings
        self.deviation_probability = float(deviation_probability)

    def estimate(self, message: StateMessage, paths_by_id: dict[int, list[Point]], actions: list[IssuedPathAction], current_step: int) -> EMCStartEstimate:
        belief: dict[Point, float] = {message.position: 1.0}
        projection_mass = gap_mass = remaining_gap_mass = 0.0
        reference_mass = 0.0
        for action in actions:
            if action.timestamp <= message.timestamp or action.timestamp > current_step:
                continue
            path = paths_by_id.get(action.path_id)
            if path is None:
                continue
            reference = path[action.path_index]
            transition = self.transition(belief, reference, action.explicit_hover)
            belief = dict(transition.belief)
            if not action.explicit_hover:
                action_mass = sum(belief.values())
                reference_mass += action_mass
                projection_mass += transition.reference_projection_rate * action_mass
                gap_mass += transition.mean_reference_gap * action_mass
                remaining_gap_mass += transition.mean_remaining_gap_after_projection * action_mass
        map_position = max(belief, key=lambda point: (belief[point], -point[1], -point[0]))
        return EMCStartEstimate(
            position=map_position,
            distribution=tuple(sorted(belief.items())),
            reference_projection_rate=projection_mass / max(reference_mass, 1e-12),
            mean_reference_gap=gap_mass / max(reference_mass, 1e-12),
            mean_remaining_gap_after_projection=remaining_gap_mass / max(reference_mass, 1e-12),
        )

    def transition(self, belief: dict[Point, float], reference: Point, explicit_hover: bool) -> EMCTransition:
        """Apply exactly one deployed reference-tracker update to a belief."""
        if explicit_hover:
            return EMCTransition(tuple(sorted(belief.items())), 0.0, 0.0, 0.0)
        updated: dict[Point, float] = {}
        projection_mass = gap_mass = remaining_gap_mass = 0.0
        for position, probability in belief.items():
            projected = reference_projection(self.buildings, position, reference)
            updated[position] = updated.get(position, 0.0) + probability * self.deviation_probability
            updated[projected] = updated.get(projected, 0.0) + probability * (1.0 - self.deviation_probability)
            gap_mass += probability * hypot(position[0] - reference[0], position[1] - reference[1])
            remaining_gap_mass += probability * hypot(projected[0] - reference[0], projected[1] - reference[1])
            if projected != reference:
                projection_mass += probability
        if abs(sum(updated.values()) - 1.0) > 1e-12:
            raise AssertionError("EMC belief probability mass is not conserved")
        return EMCTransition(tuple(sorted(updated.items())), projection_mass, gap_mass, remaining_gap_mass)


class DeterministicReplayEstimator:
    """Single-state replay using the executor's graph-legal reference tracker.

    Action timestamps denote the end of the commanded step. No execution
    outcomes or current vehicle state are available to this estimator.
    """

    physical_anchor_belief = True

    def __init__(self, buildings: np.ndarray):
        self.buildings = buildings

    def estimate(self, message: StateMessage, paths_by_id: dict[int, list[Point]], actions: list[IssuedPathAction], current_step: int) -> EMCStartEstimate:
        position = message.position
        if not legal_cell(self.buildings, position):
            raise ValueError("delivered position is outside the legal graph")
        for action in actions:
            if not message.timestamp < action.timestamp <= current_step or action.explicit_hover:
                continue
            path = paths_by_id.get(action.path_id)
            if path is not None:
                position = reference_projection(self.buildings, position, path[action.path_index])
        return EMCStartEstimate(position, ((position, 1.0),), 0.0, 0.0, 0.0)


class PlanningFailure(RuntimeError):
    """No valid joint start or path family is available under supplied evidence."""


def select_joint_starts(distributions, buildings: np.ndarray) -> list[Point]:
    """Choose mutually distinct supported vertices with maximum product mass.

    A feasible vector of independent MAP estimates is already optimal and is
    returned directly. Otherwise solve a finite assignment over positive,
    graph-valid support only. Failure never introduces an unsupported state.
    """
    beliefs = []
    for distribution in distributions:
        items = distribution.items() if isinstance(distribution, dict) else distribution
        belief = {}
        for point, mass in items:
            point = tuple(point)
            if np.isfinite(mass) and mass > 0.0 and legal_cell(buildings, point):
                belief[point] = belief.get(point, 0.0) + float(mass)
        if not belief:
            raise PlanningFailure("an agent has no positive graph-valid support")
        beliefs.append(belief)
    if not beliefs:
        return []
    # Same deterministic world-coordinate tie order as RCSR's marginal MAP.
    marginal = [max(b, key=lambda p: (b[p], p[1], p[0])) for b in beliefs]
    if len(set(marginal)) == len(marginal):
        return marginal
    vertices = sorted(set().union(*(set(b) for b in beliefs)), key=lambda p: (-p[1], -p[0]))
    if len(vertices) < len(beliefs):
        raise PlanningFailure("joint supports admit no distinct start assignment")
    from scipy.optimize import linear_sum_assignment
    cost = np.full((len(beliefs), len(vertices)), np.inf)
    for i, belief in enumerate(beliefs):
        for j, point in enumerate(vertices):
            if point in belief:
                cost[i, j] = -np.log(belief[point])
    try:
        agents, selected = linear_sum_assignment(cost)
    except ValueError as exc:
        raise PlanningFailure("joint supports admit no distinct start assignment") from exc
    if len(agents) != len(beliefs) or not np.isfinite(cost[agents, selected]).all():
        raise PlanningFailure("joint supports admit no distinct start assignment")
    result = [None] * len(beliefs)
    for i, j in zip(agents, selected):
        result[i] = vertices[j]
    return result


@dataclass(frozen=True)
class ReservationConflict:
    conflict: bool
    owner_uav_ids: frozenset[int]
    conflict_type: str
    time_step: int


class ReservationTable:
    """Time-indexed vertex and segment reservations for one planning window."""

    def __init__(self, forbid_following: bool = False) -> None:
        self.forbid_following = bool(forbid_following)
        self.vertices: dict[int, dict[Point, set[int]]] = {}
        self.segments: dict[int, list[tuple[Point, Point, int, str]]] = {}

    def reserve(self, path: list[Point], owner_uav_id: int = -1, reservation_type: str = "node") -> None:
        for time, point in enumerate(path):
            self.vertices.setdefault(time, {}).setdefault(point, set()).add(owner_uav_id)
            if time:
                self.segments.setdefault(time, []).append((path[time - 1], point, owner_uav_id, reservation_type))

    def reserve_tube(self, tube: list[set[Point]], owner_uav_id: int = -1) -> None:
        for time, cells in enumerate(tube):
            bucket = self.vertices.setdefault(time, {})
            for cell in cells:
                bucket.setdefault(cell, set()).add(owner_uav_id)

    def conflicts(self, start: Point, end: Point, time: int) -> bool:
        return self.conflict_details(start, end, time).conflict

    def conflicts_cells(self, cells: set[Point], time: int) -> bool:
        return self.cell_conflict_details(cells, time).conflict

    def conflict_details(self, start: Point, end: Point, time: int) -> ReservationConflict:
        # Match executor's no-entry-to-start-occupied rule symmetrically.
        # A candidate cannot enter a prior agent's previous vertex; nor may
        # its previous vertex be the prior agent's new destination.
        owners = set(self.vertices.get(time, {}).get(end, set()))
        if self.forbid_following:
            if start != end:
                owners.update(self.vertices.get(time - 1, {}).get(end, set()))
            owners.update(self.vertices.get(time, {}).get(start, set()))
        if owners:
            return ReservationConflict(True, frozenset(owners), "node_or_tube", time)
        for other_start, other_end, owner, reservation_type in self.segments.get(time, []):
            if other_start == end and other_end == start:
                return ReservationConflict(True, frozenset({owner}), "exchange_segment", time)
            if _diagonals_cross(start, end, other_start, other_end):
                return ReservationConflict(True, frozenset({owner}), "crossing_segment", time)
        return ReservationConflict(False, frozenset(), "none", time)

    def cell_conflict_details(self, cells: set[Point], time: int) -> ReservationConflict:
        owners: set[int] = set()
        for cell in cells:
            owners.update(self.vertices.get(time, {}).get(cell, set()))
        return ReservationConflict(bool(owners), frozenset(owners), "tube", time)


def _diagonals_cross(a: Point, b: Point, c: Point, d: Point) -> bool:
    if a == b or c == d:
        return False
    return (
        abs(a[0] - b[0]) == abs(a[1] - b[1]) == 1
        and abs(c[0] - d[0]) == abs(c[1] - d[1]) == 1
        and a[0] + b[0] == c[0] + d[0]
        and a[1] + b[1] == c[1] + d[1]
    )


def validate_joint_paths(buildings: np.ndarray, paths: list[list[Point]], starts: list[Point] | None = None, horizon: int | None = None, forbid_following: bool = False) -> bool:
    """Check the complete finite-window family, including time zero and holds."""
    if not paths:
        return starts is None or len(starts) == 0
    if any(not path for path in paths) or len({len(path) for path in paths}) != 1:
        return False
    if horizon is not None and any(len(path) != horizon + 1 for path in paths):
        return False
    if starts is not None and (len(paths) != len(starts) or any(path[0] != p for path, p in zip(paths, starts))):
        return False
    reservations = ReservationTable(forbid_following=forbid_following)
    for owner, path in enumerate(paths):
        if not legal_cell(buildings, path[0]) or reservations.conflicts_cells({path[0]}, 0):
            return False
        for time in range(1, len(path)):
            if not legal_transition(buildings, path[time - 1], path[time]):
                return False
            if reservations.conflicts(path[time - 1], path[time], time):
                return False
        reservations.reserve(path, owner)
    return True


@lru_cache(maxsize=8)
def _motion_adjacency(shape: tuple[int, int], mask_bytes: bytes):
    """Content-keyed sparse graph with exactly the shared legal transitions."""
    from scipy.sparse import csr_matrix
    blocked = np.frombuffer(mask_bytes, dtype=np.bool_).reshape(shape)
    height, width = shape
    y, x = np.indices(shape)
    sources, destinations = [], []
    for dx, dy in TRACKING_MOVES[1:]:
        nx, ny = x + dx, y + dy
        valid = (nx >= 0) & (nx < width) & (ny >= 0) & (ny < height) & ~blocked
        sy, sx = y[valid], x[valid]
        ty, tx = sy + dy, sx + dx
        clear = ~blocked[ty, tx]
        if dx and dy:
            clear &= ~blocked[sy, tx] & ~blocked[ty, sx]
        sources.append((sy[clear] * width + sx[clear]).ravel())
        destinations.append((ty[clear] * width + tx[clear]).ravel())
    rows, cols = np.concatenate(sources), np.concatenate(destinations)
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(height*width, height*width))


@lru_cache(maxsize=128)
def _goal_distance_map(shape: tuple[int, int], mask_bytes: bytes, goal: Point, goal_radius: float) -> np.ndarray:
    """Exact service-region distances, computed on a cached legal graph."""
    from scipy.sparse.csgraph import dijkstra
    buildings = np.frombuffer(mask_bytes, dtype=np.bool_).reshape(shape)
    y, x = np.indices(shape)
    service = (~buildings) & (np.hypot(x-goal[0], y-goal[1]) <= goal_radius)
    indices = np.flatnonzero(service)
    if len(indices):
        distances = dijkstra(_motion_adjacency(shape, mask_bytes), directed=False,
                             indices=indices, unweighted=True, min_only=True).reshape(shape)
    else:
        distances = np.full(shape, np.inf)
    distances.flags.writeable = False
    return distances


class ReferenceSpaceTimePrioritizedPlanner:
    """Single deterministic space-time solver used by RHCR-adapted."""

    moves = ((0, 0), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))

    def __init__(self, buildings: np.ndarray, goal_radius: float):
        self.buildings = buildings
        self.height, self.width = buildings.shape
        self.goal_radius = float(goal_radius)

    def path(self, start: Point, goal: Point, window: int, reservations: ReservationTable, tube_buffer: int = 0, blocker_collector=None) -> list[Point] | None:
        if not self._free(start) or reservations.conflicts_cells(self._tube_cells(start, tube_buffer), 0):
            return None
        self._distance_map = _goal_distance_map(
            self.buildings.shape, np.asarray(self.buildings, dtype=np.bool_).tobytes(), goal, self.goal_radius
        )
        queue: deque[tuple[Point, int]] = deque([(start, 0)])
        parent: dict[tuple[Point, int], tuple[Point, int] | None] = {(start, 0): None}
        best = None
        while queue:
            point, time = queue.popleft()
            # Early termination is legal only if every padded hold is reserved-safe.
            if self._goal_reached(point, goal) and all(
                not reservations.conflicts(point, point, t)
                and not reservations.conflicts_cells(self._tube_cells(point, tube_buffer), t)
                for t in range(time + 1, window + 1)
            ):
                return self._finish(self._reconstruct((point, time), parent), window)
            if time == window:
                if best is None or self._goal_key(point, goal) < self._goal_key(best[0], goal):
                    best = (point, time)
                continue
            nonwait_rejected: set[int] = set()
            nonwait_accepted = False
            wait_accepted = False
            for dx, dy in self.moves:
                candidate = (point[0] + dx, point[1] + dy)
                key = (candidate, time + 1)
                if key in parent or not legal_transition(self.buildings, point, candidate):
                    continue
                conflict = reservations.conflict_details(point, candidate, time + 1)
                tube_conflict = reservations.cell_conflict_details(self._tube_cells(candidate, tube_buffer), time + 1)
                owners = conflict.owner_uav_ids | tube_conflict.owner_uav_ids
                if owners:
                    if blocker_collector is not None:
                        blocker_collector.record(owners, "hard_infeasibility_blocker")
                    if dx or dy:
                        nonwait_rejected.update(owners)
                    continue
                parent[key] = (point, time)
                queue.append(key)
                if dx or dy:
                    nonwait_accepted = True
                else:
                    wait_accepted = True
            if blocker_collector is not None and wait_accepted and not nonwait_accepted and nonwait_rejected:
                blocker_collector.record(nonwait_rejected, "forced_wait_blocker")
        if best is None:
            return None
        return self._finish(self._reconstruct(best, parent), window)

    def _free(self, point: Point) -> bool:
        return legal_cell(self.buildings, point)

    def _goal_reached(self, point: Point, goal: Point) -> bool:
        return hypot(point[0] - goal[0], point[1] - goal[1]) <= self.goal_radius

    def _tube_cells(self, point: Point, buffer: int) -> set[Point]:
        if buffer <= 0:
            return {point}
        return {(point[0] + dx, point[1] + dy) for dx in range(-buffer, buffer + 1) for dy in range(-buffer, buffer + 1) if hypot(dx, dy) <= buffer and self._free((point[0] + dx, point[1] + dy))}

    def _goal_key(self, point: Point, goal: Point) -> tuple[float, float, int, int]:
        distances = self._distance_map
        return (float(distances[point[1], point[0]]), hypot(point[0] - goal[0], point[1] - goal[1]), point[1], point[0])

    @staticmethod
    def _reconstruct(last: tuple[Point, int], parent: dict[tuple[Point, int], tuple[Point, int] | None]) -> list[Point]:
        path: list[Point] = []
        current: tuple[Point, int] | None = last
        while current is not None:
            path.append(current[0])
            current = parent[current]
        return list(reversed(path))

    @staticmethod
    def _finish(path: list[Point], window: int) -> list[Point]:
        return path + [path[-1]] * (window + 1 - len(path))


@lru_cache(maxsize=8)
def _planner_successor_cache(shape, mask_bytes, moves):
    """Bounded content-keyed lazy geometry; no search or reservation state."""
    return {}


class SpaceTimePrioritizedPlanner(ReferenceSpaceTimePrioritizedPlanner):
    """Exact shared BFS with cached geometry and allocation-free conflict queries.

    The reference implementation above remains available for differential tests.
    """

    def path(self, start: Point, goal: Point, window: int, reservations: ReservationTable, tube_buffer: int = 0, blocker_collector=None) -> list[Point] | None:
        if (type(reservations) is not ReservationTable
                or type(self)._tube_cells is not ReferenceSpaceTimePrioritizedPlanner._tube_cells):
            return super().path(start, goal, window, reservations, tube_buffer, blocker_collector)
        if not self._free(start) or reservations.conflicts_cells(self._tube_cells(start, tube_buffer), 0):
            return None
        mask_bytes = np.asarray(self.buildings, dtype=np.bool_).tobytes()
        self._distance_map = _goal_distance_map(
            self.buildings.shape, mask_bytes, goal, self.goal_radius
        )
        # The content key observes map mutations between path calls. Geometry
        # is shared across controllers, never search/reservation/goal state.
        successors = _planner_successor_cache(self.buildings.shape, mask_bytes, tuple(self.moves))
        tubes = {}
        def tube(point):
            if point not in tubes:
                tubes[point] = self._tube_cells(point, tube_buffer)
            return tubes[point]

        # Compile segment conflicts in reservation insertion order. Node owners
        # still take precedence and segments still return only the first owner.
        segments = {}
        for t, entries in reservations.segments.items():
            index = segments[t] = {}
            for a, b, owner, kind in entries:
                index.setdefault((b, a), frozenset((owner,)))
                if abs(a[0]-b[0]) == abs(a[1]-b[1]) == 1:
                    index.setdefault((a, b), frozenset((owner,)))
                    c, d = (a[0], b[1]), (b[0], a[1])
                    index.setdefault((c, d), frozenset((owner,)))
                    index.setdefault((d, c), frozenset((owner,)))
        def conflict_owners(point, candidate, t):
            vertices = reservations.vertices.get(t, {})
            owners = set(vertices.get(candidate, ()))
            if reservations.forbid_following:
                if point != candidate:
                    owners.update(reservations.vertices.get(t-1, {}).get(candidate, ()))
                owners.update(vertices.get(point, ()))
            if not owners:
                owners.update(segments.get(t, {}).get((point, candidate), ()))
            if tube_buffer > 0:
                for cell in tube(candidate):
                    owners.update(vertices.get(cell, ()))
            # buffer <= 0's tube is exactly {candidate}, already included above.
            return owners

        queue: deque[tuple[Point, int]] = deque([(start, 0)])
        parent: dict[tuple[Point, int], tuple[Point, int] | None] = {(start, 0): None}
        best = None
        while queue:
            point, time = queue.popleft()
            # Early termination is legal only if every padded hold is reserved-safe.
            if self._goal_reached(point, goal) and all(
                not reservations.conflicts(point, point, t)
                and not reservations.conflicts_cells(tube(point), t)
                for t in range(time + 1, window + 1)
            ):
                return self._finish(self._reconstruct((point, time), parent), window)
            if time == window:
                if best is None or self._goal_key(point, goal) < self._goal_key(best[0], goal):
                    best = (point, time)
                continue
            nonwait_rejected: set[int] = set()
            nonwait_accepted = False
            wait_accepted = False
            if point not in successors:
                successors[point] = tuple(
                    ((point[0] + dx, point[1] + dy), bool(dx or dy))
                    for dx, dy in self.moves
                    if legal_transition(self.buildings, point, (point[0] + dx, point[1] + dy))
                )
            for candidate, moving in successors[point]:
                key = (candidate, time + 1)
                if key in parent:
                    continue
                owners = conflict_owners(point, candidate, time + 1)
                if owners:
                    if blocker_collector is not None:
                        blocker_collector.record(frozenset(owners), "hard_infeasibility_blocker")
                    if moving:
                        nonwait_rejected.update(owners)
                    continue
                parent[key] = (point, time)
                queue.append(key)
                if moving:
                    nonwait_accepted = True
                else:
                    wait_accepted = True
            if blocker_collector is not None and wait_accepted and not nonwait_accepted and nonwait_rejected:
                blocker_collector.record(nonwait_rejected, "forced_wait_blocker")
        if best is None:
            return None
        return self._finish(self._reconstruct(best, parent), window)


class RHCRAdapted:
    """Global replanning reference adapted to the public-target city map."""

    def __init__(self, config: RHCRConfig = RHCRConfig(), forbid_following: bool = False):
        self.config = config
        self.forbid_following = bool(forbid_following)
        self.last_plan_fallback = False
        self.last_failure_reason = ""

    @staticmethod
    def _check_starts(buildings: np.ndarray, positions: list[Point]) -> None:
        if len(set(positions)) != len(positions):
            raise PlanningFailure("planning starts must be mutually distinct")
        if any(not legal_cell(buildings, point) for point in positions):
            raise PlanningFailure("planning starts must belong to the legal graph")

    def _wait_fallback(self, buildings: np.ndarray, positions: list[Point], reason: str) -> list[list[Point]]:
        paths = [[position] * (self.config.planning_window + 1) for position in positions]
        if not validate_joint_paths(buildings, paths, positions, self.config.planning_window, forbid_following=self.forbid_following):
            raise PlanningFailure("no validated wait fallback exists")
        self.last_plan_fallback = True
        self.last_failure_reason = reason
        return paths

    def plan(self, buildings: np.ndarray, positions: list[Point], goals: list[Point]) -> list[list[Point]]:
        if len(positions) != len(goals):
            raise ValueError("positions and goals must have identical length")
        self._check_starts(buildings, positions)
        self.last_plan_fallback = False
        self.last_failure_reason = ""
        reservations = ReservationTable(forbid_following=self.forbid_following)
        planner = SpaceTimePrioritizedPlanner(buildings, goal_radius=1.0)
        paths: list[list[Point]] = []
        for owner, (position, goal) in enumerate(zip(positions, goals)):
            # Starts of agents not yet planned occupy time zero. They are not
            # assumed stationary for the full horizon.
            reservations.vertices[0] = {
                point: {index} for index, point in enumerate(positions) if index != owner
            }
            path = planner.path(position, goal, self.config.planning_window, reservations)
            if path is None:
                return self._wait_fallback(buildings, positions, "prioritized search failed")
            reservations.reserve(path, owner)
            paths.append(path)
        if not validate_joint_paths(buildings, paths, positions, self.config.planning_window, forbid_following=self.forbid_following):
            return self._wait_fallback(buildings, positions, "joint path validation failed")
        return paths

    def plan_with_tubes(self, buildings: np.ndarray, positions: list[Point], goals: list[Point], nominal_paths: list[list[Point]], buffers: list[int]) -> list[list[Point]]:
        """Global prioritized RHCR with other UAVs' AoI tubes as reservations.

        For UAV i its own previous tube is intentionally absent; after i is
        replanned, its new tube is immediately visible to all later UAVs.
        """
        self._check_starts(buildings, positions)
        self.last_plan_fallback = False
        self.last_failure_reason = ""
        paths = [list(path) for path in nominal_paths]
        for index, (position, goal) in enumerate(zip(positions, goals)):
            reservations = ReservationTable(forbid_following=self.forbid_following)
            for other in range(len(paths)):
                if other == index:
                    continue
                reservations.reserve_tube(occupancy_tube(buildings, paths[other], buffers[other]), other)
            planner = SpaceTimePrioritizedPlanner(buildings, goal_radius=1.0)
            path = planner.path(position, goal, self.config.planning_window, reservations, tube_buffer=buffers[index])
            if path is None:
                path = paths[index]
            paths[index] = path
        if not validate_joint_paths(buildings, paths, positions, self.config.planning_window, forbid_following=self.forbid_following):
            return self._wait_fallback(buildings, positions, "tube replanning did not yield a valid joint family")
        return paths
