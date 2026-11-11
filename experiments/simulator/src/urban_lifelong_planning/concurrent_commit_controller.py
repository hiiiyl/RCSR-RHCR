"""Information-restricted controller for Concurrent-Commit RHCR."""
from __future__ import annotations

import numpy as np

from .concurrent_commit_planner import ConcurrentCommitPlanner
from .concurrent_commit_state import CommitBatch, ConcurrentCommitState, Point
from .delayed_state_channel import StateMessage
from .rhcr_adapted import RHCRConfig


class ConcurrentCommitController:
    """Uses delivered messages, map/goals, and its own issued command ledger only."""

    def __init__(self, drone_count: int, config: RHCRConfig, commit_window: int | None = None):
        self.state = ConcurrentCommitState(drone_count, commit_window=config.execution_horizon if commit_window is None else commit_window)
        if config.planning_window < self.state.commit_window:
            raise ValueError("planning_window must cover the complete commit window")
        self.planner = ConcurrentCommitPlanner(config)

    def synchronize(self, messages: list[StateMessage]) -> int:
        return self.state.synchronize_delivered_states(messages)

    def planning_starts(self) -> tuple[Point, ...]:
        return self.state.planning_starts()

    def replan(self, buildings: np.ndarray, messages: list[StateMessage], goals: list[Point]) -> tuple[CommitBatch, int]:
        reanchored = self.synchronize(messages)
        return self.planner.plan(buildings, self.state, goals), reanchored

    def dispatch_next(self) -> tuple[int, tuple[Point, ...]]:
        return self.state.dispatch_next()
