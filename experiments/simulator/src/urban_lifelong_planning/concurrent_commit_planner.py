"""CC-RHCR planning wrapper around the project's frozen space-time planner."""
from __future__ import annotations

import numpy as np

from .concurrent_commit_state import ConcurrentCommitState, CommitBatch, Point
from .rhcr_adapted import RHCRAdapted, RHCRConfig


class ConcurrentCommitPlanner:
    """Plans a replaceable suffix while preserving the configured commit prefix."""

    def __init__(self, config: RHCRConfig):
        self.config = config
        self._planner = RHCRAdapted(config)

    def plan(self, buildings: np.ndarray, state: ConcurrentCommitState, goals: list[Point]) -> CommitBatch:
        starts = list(state.planning_starts())
        if len(goals) != len(starts):
            raise ValueError("goals and CC-RHCR planning starts must have equal length")
        if state.commit_window > self.config.planning_window:
            raise ValueError("planning_window must cover commit_window")
        paths = self._planner.plan(buildings, starts, goals)
        return state.install_plan(paths)
