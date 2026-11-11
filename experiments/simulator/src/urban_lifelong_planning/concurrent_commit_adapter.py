"""Environment-facing CC-RHCR adapter with no access to live execution state."""
from __future__ import annotations

import numpy as np

from .concurrent_commit_controller import ConcurrentCommitController
from .concurrent_commit_state import CommitBatch, Point
from .delayed_state_channel import StateMessage
from .rhcr_adapted import RHCRConfig


class ConcurrentCommitAdapter:
    """Bridges delayed telemetry to the concurrent-commit controller."""

    method_name = "CC-RHCR"

    def __init__(self, drone_count: int, config: RHCRConfig, commit_window: int | None = None):
        self.controller = ConcurrentCommitController(drone_count, config, commit_window=commit_window)
        self.last_batch: CommitBatch | None = None
        self.last_reanchor_count = 0

    def synchronize(self, messages: list[StateMessage]) -> int:
        self.last_reanchor_count = self.controller.synchronize(messages)
        return self.last_reanchor_count

    def planning_starts(self) -> tuple[Point, ...]:
        return self.controller.planning_starts()

    def plan(self, buildings: np.ndarray, messages: list[StateMessage], goals: list[Point]) -> CommitBatch:
        batch, reanchored = self.controller.replan(buildings, messages, goals)
        self.last_batch = batch
        self.last_reanchor_count += reanchored
        return batch

    def dispatch_next(self) -> tuple[int, tuple[Point, ...]]:
        return self.controller.dispatch_next()
