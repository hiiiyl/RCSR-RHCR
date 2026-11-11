"""Conflict-component-only repair with a global RHCR fallback."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .aoi_occupancy_tube import occupancy_tube
from .conflict_graph import ConflictGraph, build_conflict_graph
from .rhcr_adapted import RHCRConfig, ReservationTable, SpaceTimePrioritizedPlanner

Point = tuple[int, int]


@dataclass
class RepairResult:
    paths: list[list[Point]]
    graph: ConflictGraph
    repaired_components: list[tuple[int, ...]]
    fallback: bool
    planning_time_ms: float
    closure_sets: list[tuple[int, ...]]
    initial_component_sizes: list[int]
    blocker_uav_ids: list[tuple[int, ...]]
    hard_infeasibility_blocker_count: int
    forced_wait_blocker_count: int


class _BlockerCollector:
    def __init__(self, fixed_uavs: set[int]):
        self.fixed_uavs = fixed_uavs
        self.owners: set[int] = set()
        self.hard = 0
        self.forced_wait = 0

    def record(self, owners: set[int] | frozenset[int], kind: str) -> None:
        relevant = set(owners) & self.fixed_uavs
        if not relevant:
            return
        self.owners.update(relevant)
        if kind == "forced_wait_blocker":
            self.forced_wait += len(relevant)
        else:
            self.hard += len(relevant)


class LocalConflictRepair:
    def __init__(self, config: RHCRConfig):
        self.config = config

    def repair(self, buildings: np.ndarray, paths: list[list[Point]], goals: list[Point], tubes: list[list[set[Point]]], buffers: list[int] | None = None) -> RepairResult:
        started = perf_counter()
        graph = build_conflict_graph(paths, tubes)
        components = graph.components()
        repaired = [list(path) for path in paths]
        buffers = [0] * len(paths) if buffers is None else buffers
        closure_sets: list[tuple[int, ...]] = []
        blocker_sets: list[tuple[int, ...]] = []
        hard_count = forced_wait_count = 0
        for component in components:
            repair_set = set(component)
            while True:
                fixed = set(range(len(paths))) - repair_set
                reservations = ReservationTable()
                for index in fixed:
                    reservations.reserve(repaired[index], index)
                    reservations.reserve_tube(occupancy_tube(buildings, repaired[index], buffers[index]), index)
                collector = _BlockerCollector(fixed)
                candidate = [list(path) for path in repaired]
                planner = SpaceTimePrioritizedPlanner(buildings, goal_radius=1.0)
                feasible = True
                for index in sorted(repair_set):
                    path = planner.path(paths[index][0], goals[index], self.config.planning_window, reservations, tube_buffer=buffers[index], blocker_collector=collector)
                    if path is None:
                        feasible = False
                        break
                    candidate[index] = path
                    reservations.reserve(path, index)
                    reservations.reserve_tube(occupancy_tube(buildings, path, buffers[index]), index)
                hard_count += collector.hard
                forced_wait_count += collector.forced_wait
                new_blockers = collector.owners - repair_set
                if not feasible and not new_blockers:
                    return RepairResult(paths, graph, components, True, (perf_counter() - started) * 1000.0, closure_sets, [len(c) for c in components], blocker_sets, hard_count, forced_wait_count)
                if new_blockers:
                    repair_set.update(new_blockers)
                    continue
                repaired = candidate
                closure_sets.append(tuple(sorted(repair_set)))
                blocker_sets.append(tuple(sorted(collector.owners)))
                break
        return RepairResult(repaired, graph, components, False, (perf_counter() - started) * 1000.0, closure_sets, [len(c) for c in components], blocker_sets, hard_count, forced_wait_count)
