"""Controller-side commit ledger for the CC-RHCR adapted baseline.

The ledger deliberately contains only delivered telemetry and controller-issued
commands.  It has no reference to the environment's live state or execution
outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass

from .delayed_state_channel import StateMessage

Point = tuple[int, int]


@dataclass(frozen=True)
class CommitBatch:
    """One immutable command prefix and its replaceable future suffix."""

    active_plan_id: int
    commit_generation: int
    committed_prefix: tuple[tuple[Point, ...], ...]
    uncommitted_suffix: tuple[tuple[Point, ...], ...]
    nominal_commit_end: tuple[Point, ...]
    planning_starts: tuple[Point, ...]


class ConcurrentCommitState:
    """Tracks delayed anchors and a fixed-length controller command commit."""

    def __init__(self, drone_count: int, commit_window: int = 2):
        if int(drone_count) <= 0:
            raise ValueError("drone_count must be positive")
        if int(commit_window) < 1 or int(commit_window) != commit_window:
            raise ValueError("commit_window must be a positive integer")
        self.drone_count = int(drone_count)
        self.commit_window = int(commit_window)
        self.last_delivered_state: list[StateMessage | None] = [None] * self.drone_count
        self._nominal_positions: list[Point | None] = [None] * self.drone_count
        self._committed_prefix: list[tuple[Point, ...]] = [tuple() for _ in range(self.drone_count)]
        self._uncommitted_suffix: list[tuple[Point, ...]] = [tuple() for _ in range(self.drone_count)]
        self._commit_cursor = 0
        self.active_plan_id = -1
        self.commit_generation = 0
        self.reanchor_count = 0

    @property
    def committed_prefix(self) -> tuple[tuple[Point, ...], ...]:
        return tuple(self._committed_prefix)

    @property
    def uncommitted_suffix(self) -> tuple[tuple[Point, ...], ...]:
        return tuple(self._uncommitted_suffix)

    @property
    def committed_actions_remaining(self) -> int:
        return self.commit_window - self._commit_cursor if self.active_plan_id >= 0 else 0

    def synchronize_delivered_states(self, messages: list[StateMessage]) -> int:
        """Anchor only to newer delivered packets; never to a live state."""
        if len(messages) != self.drone_count:
            raise ValueError("every CC-RHCR update needs one message per UAV")
        reanchored = 0
        for message in messages:
            prior = self.last_delivered_state[message.drone_id]
            if prior is None or message.timestamp > prior.timestamp:
                self.last_delivered_state[message.drone_id] = message
                self._nominal_positions[message.drone_id] = message.position
                reanchored += 1
        self.reanchor_count += reanchored
        return reanchored

    def planning_starts(self) -> tuple[Point, ...]:
        if any(position is None for position in self._nominal_positions):
            raise RuntimeError("CC-RHCR needs an initial delivered state for every UAV")
        return tuple(position for position in self._nominal_positions if position is not None)

    def install_plan(self, paths: list[list[Point]]) -> CommitBatch:
        """Commit the configured prefix; replace only the future suffix."""
        if self.committed_actions_remaining:
            raise RuntimeError("CC-RHCR cannot replace an active committed prefix")
        if len(paths) != self.drone_count or any(len(path) < self.commit_window + 1 for path in paths):
            raise ValueError("planner paths must include a start plus the committed actions")
        starts = self.planning_starts()
        if any(path[0] != starts[index] for index, path in enumerate(paths)):
            raise AssertionError("CC-RHCR planner path must start at the nominal controller state")
        self.active_plan_id += 1
        self.commit_generation += 1
        self._commit_cursor = 0
        self._committed_prefix = [tuple(path[1:self.commit_window + 1]) for path in paths]
        self._uncommitted_suffix = [tuple(path[self.commit_window + 1:]) for path in paths]
        self._nominal_positions = [path[self.commit_window] for path in paths]
        return CommitBatch(
            active_plan_id=self.active_plan_id,
            commit_generation=self.commit_generation,
            committed_prefix=self.committed_prefix,
            uncommitted_suffix=self.uncommitted_suffix,
            nominal_commit_end=tuple(path[self.commit_window] for path in paths),
            planning_starts=starts,
        )

    def dispatch_next(self) -> tuple[int, tuple[Point, ...]]:
        """Return the next immutable command without receiving execution feedback."""
        if not self.committed_actions_remaining:
            raise RuntimeError("CC-RHCR has no committed action to dispatch")
        command_index = self._commit_cursor + 1
        commands = tuple(prefix[self._commit_cursor] for prefix in self._committed_prefix)
        self._commit_cursor += 1
        return command_index, commands
