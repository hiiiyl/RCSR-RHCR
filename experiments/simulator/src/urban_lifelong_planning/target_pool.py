"""Deterministic, public, continuously replenished target pool."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Point = tuple[int, int]


@dataclass
class TargetRecord:
    target_id: int
    position: Point
    release_step: int
    assignment_step: int | None = None
    completion_step: int | None = None
    assigned_uav: int | None = None
    path_length: float = 0.0
    replan_count: int = 0

    def as_dict(self) -> dict[str, int | float | None]:
        completion = self.completion_step
        return {
            "target_id": self.target_id,
            "x": self.position[0],
            "y": self.position[1],
            "release_step": self.release_step,
            "assignment_step": self.assignment_step,
            "completion_step": completion,
            "assigned_uav": self.assigned_uav,
            "response_time": None if completion is None else completion - self.release_step,
            "path_length": self.path_length,
            "replan_count": self.replan_count,
        }


class PublicTargetPool:
    """Keeps a fixed number of unique public targets on free map cells."""

    def __init__(self, active_target_count: int, rng: np.random.Generator, allow_coincident_targets: bool = False):
        if active_target_count <= 0:
            raise ValueError("active_target_count must be positive")
        self.active_target_count = int(active_target_count)
        self.rng = rng
        self.allow_coincident_targets = bool(allow_coincident_targets)
        self.active: dict[int, TargetRecord] = {}
        self.completed: list[TargetRecord] = []
        self._next_id = 0

    def initialize(self, sampler: Callable[[set[Point]], Point], occupied: set[Point]) -> None:
        if self.active or self.completed:
            raise RuntimeError("target pool is already initialized")
        self._replenish(step=0, sampler=sampler, occupied=occupied)

    def assign(self, target_id: int, drone_idx: int, step: int) -> None:
        target = self.active[target_id]
        if target.assigned_uav is not None:
            raise RuntimeError(f"target {target_id} is already assigned")
        target.assigned_uav = int(drone_idx)
        target.assignment_step = int(step)

    def complete(
        self,
        target_id: int,
        step: int,
        sampler: Callable[[set[Point]], Point],
        occupied: set[Point],
    ) -> TargetRecord:
        target = self.active.pop(target_id)
        target.completion_step = int(step)
        self.completed.append(target)
        self._replenish(step=step, sampler=sampler, occupied=occupied)
        return target

    def add_path_length(self, target_id: int, distance: float) -> None:
        if target_id in self.active:
            self.active[target_id].path_length += float(distance)

    def pending_ids(self) -> list[int]:
        return [key for key, target in self.active.items() if target.assigned_uav is None]

    def assert_invariants(self, occupied: set[Point]) -> None:
        if len(self.active) != self.active_target_count:
            raise AssertionError("active target count changed")
        positions = [target.position for target in self.active.values()]
        if not self.allow_coincident_targets and len(positions) != len(set(positions)):
            raise AssertionError("active targets overlap")

    def _replenish(self, step: int, sampler: Callable[[set[Point]], Point], occupied: set[Point]) -> None:
        while len(self.active) < self.active_target_count:
            excluded = occupied | {target.position for target in self.active.values()}
            target = TargetRecord(self._next_id, sampler(excluded), int(step))
            self.active[target.target_id] = target
            self._next_id += 1
