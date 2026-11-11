"""Fixed-delay uplink state channel used by Stage 3 planning experiments."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

Point = tuple[int, int]


@dataclass(frozen=True)
class StateMessage:
    drone_id: int
    timestamp: int
    position: Point
    energy: float
    target_id: int | None
    path_index: int
    # The path identifier is telemetry only.  It lets diagnostic runs
    # distinguish a stale position report from an acknowledged command path.
    path_id: int = -1


DelayProvider = Callable[[int, int], int]


class DelayedStateChannel:
    """The planner can read only delivered messages, never live UAV state."""

    def __init__(self, drone_count: int, delay_steps: int, delay_provider: DelayProvider | None = None):
        if int(delay_steps) < 0:
            raise ValueError("delay_steps must be non-negative")
        self.drone_count = int(drone_count)
        self.delay_steps = int(delay_steps)
        self.delay_provider = delay_provider
        self._pending: dict[int, list[StateMessage]] = {}
        self._delivered: dict[int, StateMessage] = {}
        self.delivery_records: list[dict[str, int]] = []
        self.visibility_records: list[dict[str, int]] = []

    def bootstrap(self, messages: list[StateMessage]) -> None:
        """Initial launch state is public mission configuration, not a live update."""
        self._accept(messages, visible_step=0)

    def send(self, messages: list[StateMessage]) -> None:
        if len(messages) != self.drone_count:
            raise ValueError("every UAV must send exactly one state message")
        for message in messages:
            delay = self.delay_steps if self.delay_provider is None else int(self.delay_provider(message.timestamp, message.drone_id))
            if delay < 0:
                raise ValueError("message delay provider returned a negative delay")
            delivery_step = message.timestamp + delay
            self._pending.setdefault(delivery_step, []).append(message)
            self.delivery_records.append({
                "drone_id": message.drone_id,
                "send_step": message.timestamp,
                "delay_steps": delay,
                "delivery_step": delivery_step,
            })

    def deliver(self, step: int) -> None:
        for delivery_step in sorted(key for key in self._pending if key <= step):
            self._accept(self._pending.pop(delivery_step), visible_step=step)

    def planner_states(self, step: int) -> list[StateMessage]:
        if len(self._delivered) != self.drone_count:
            raise RuntimeError("planner has not received an initial state for every UAV")
        states = [self._delivered[index] for index in range(self.drone_count)]
        if any(message.timestamp > step for message in states):
            raise AssertionError("future state leaked into planner")
        return states

    def aoi(self, step: int) -> list[int]:
        return [step - message.timestamp for message in self.planner_states(step)]

    def _accept(self, messages: list[StateMessage], visible_step: int | None = None) -> None:
        for message in messages:
            current = self._delivered.get(message.drone_id)
            if current is None or message.timestamp >= current.timestamp:
                self._delivered[message.drone_id] = message
                if visible_step is not None:
                    self.visibility_records.append({
                        "drone_id": message.drone_id,
                        "send_step": message.timestamp,
                        "first_visible_step": int(visible_step),
                    })
