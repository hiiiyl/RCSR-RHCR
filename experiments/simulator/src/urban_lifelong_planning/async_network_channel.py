"""Asynchronous network channel with random delay, packet loss, and natural
out-of-order arrival.

This module is designed for Stage 1 (RAS extension). It shares the same
``send`` / ``deliver`` / ``planner_states`` / ``aoi`` interface as
``DelayedStateChannel`` so that existing environment methods can use it with
minimal changes.

Architecture
------------
1. A *network trace* is pre-generated for each (condition, seed) pair.  The
   trace encodes per-step per-drone ``(delay, dropped)`` tuples drawn from the
   configured distribution.
2. ``AsyncNetworkChannel`` reads the trace at construction time and replays it
   deterministically at runtime.
3. Because all three methods (RHCR-adapted, CC-RHCR, RCSR-RHCR) share the
   same trace file, network conditions are strictly paired.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

Point = tuple[int, int]

# ── trace generation ───────────────────────────────────────────────────────

def generate_async_trace(
    seed: int,
    drone_count: int,
    episode_steps: int,
    max_delay: int,
    loss_rate: float,
    condition_name: str = "",
) -> dict[str, Any]:
    """Pre-generate a deterministic network trace.

    Parameters
    ----------
    seed : int
        Base seed for the network RNG.
    drone_count : int
        Number of UAVs.
    episode_steps : int
        Number of simulation steps. Messages may be sent at steps
        0..episode_steps (inclusive) because the environment sends after
        the last execution substep.
    max_delay : int
        Maximum delay in steps. Each non-dropped message draws a delay from
        ``randint(0, max_delay)``.
    loss_rate : float
        Probability that a message is dropped.

    Returns
    -------
    dict
        Trace payload with keys ``metadata`` and ``messages``.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4E3757C1]))

    messages: dict[str, dict[str, list]] = {}
    for step in range(episode_steps + 1):
        step_messages: dict[str, list] = {}
        for drone_id in range(drone_count):
            dropped = bool(rng.random() < loss_rate)
            delay = int(rng.integers(0, max_delay + 1))
            step_messages[str(drone_id)] = [delay, dropped]
        messages[str(step)] = step_messages

    actual_delays = []
    actual_dropped = 0
    total = 0
    for step_data in messages.values():
        for drone_id_str, (delay, dropped) in step_data.items():
            total += 1
            if dropped:
                actual_dropped += 1
            else:
                actual_delays.append(delay)

    return {
        "metadata": {
            "condition": condition_name,
            "seed": seed,
            "drone_count": drone_count,
            "episode_steps": episode_steps,
            "max_delay": max_delay,
            "loss_rate": loss_rate,
            "actual_mean_delay": float(np.mean(actual_delays)) if actual_delays else 0.0,
            "actual_p90_delay": float(np.percentile(actual_delays, 90)) if actual_delays else 0.0,
            "actual_loss_rate": actual_dropped / total if total > 0 else 0.0,
        },
        "messages": messages,
    }


def save_async_trace(trace: dict[str, Any], path: Path) -> None:
    """Write a trace to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def load_async_trace(path: Path) -> dict[str, Any]:
    """Load a trace from a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


# ── channel ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AsyncStateMessage:
    """State message matching the StateMessage interface from delayed_state_channel."""

    drone_id: int
    timestamp: int
    position: Point
    energy: float
    target_id: int | None
    path_index: int
    path_id: int = -1


class AsyncNetworkChannel:
    """Delayed-state channel backed by a pre-generated asynchronous network trace.

    The planner can read only delivered messages, never live UAV state.
    """

    def __init__(self, drone_count: int, trace: dict[str, Any]):
        self.drone_count = int(drone_count)
        self._trace = trace
        self._pending: dict[int, list[AsyncStateMessage]] = {}
        self._delivered: dict[int, AsyncStateMessage] = {}
        self.delivery_records: list[dict[str, int]] = []
        self.visibility_records: list[dict[str, int]] = []
        self._trace_messages = trace["messages"]
        self._out_of_order_count = 0
        self._total_delivered_count = 0
        self._latest_delivered_timestamp: dict[int, int] = {}

    # ── public API (matches DelayedStateChannel) ─────────────────────────

    def bootstrap(self, messages: list) -> None:
        """Initial launch state is public mission configuration, not a live update."""
        self._accept(messages, visible_step=0)

    def send(self, messages: list) -> None:
        """Enqueue messages using delays from the pre-generated trace.

        Accepts any message object with ``drone_id``, ``timestamp``,
        ``position``, ``energy``, ``target_id``, ``path_index``, ``path_id``
        attributes (duck-typed for compatibility with both ``StateMessage``
        and ``AsyncStateMessage``).

        At t=0, messages are delivered immediately (no delay, no loss).
        """
        if len(messages) != self.drone_count:
            raise ValueError("every UAV must send exactly one state message")
        for message in messages:
            step = message.timestamp
            drone_id = message.drone_id
            # t=0 is always delivered immediately
            if step == 0:
                self._accept([message], visible_step=0)
                continue
            # Look up delay/drop from trace
            step_str = str(step)
            drone_str = str(drone_id)
            try:
                delay, dropped = self._trace_messages[step_str][drone_str]
            except KeyError:
                raise RuntimeError(
                    f"trace missing entry for step={step} drone={drone_id}"
                ) from None
            if dropped:
                # Message is permanently lost
                self.delivery_records.append({
                    "drone_id": drone_id,
                    "send_step": step,
                    "delay_steps": -1,
                    "delivery_step": -1,
                    "dropped": 1,
                })
                continue
            delivery_step = step + delay
            self._pending.setdefault(delivery_step, []).append(message)
            self.delivery_records.append({
                "drone_id": drone_id,
                "send_step": step,
                "delay_steps": delay,
                "delivery_step": delivery_step,
                "dropped": 0,
            })

    def deliver(self, step: int) -> None:
        """Make all messages whose delivery_step <= step visible."""
        for delivery_step in sorted(key for key in self._pending if key <= step):
            self._accept(self._pending.pop(delivery_step), visible_step=step)

    def planner_states(self, step: int) -> list:
        """Return the currently visible state messages.

        For each drone, the message with the largest *send_timestamp* is
        selected.  This ensures that a late-arriving old message never
        overwrites a newer one (natural out-of-order handling).
        """
        if len(self._delivered) != self.drone_count:
            raise RuntimeError("planner has not received an initial state for every UAV")
        states = [self._delivered[index] for index in range(self.drone_count)]
        if any(message.timestamp > step for message in states):
            raise AssertionError("future state leaked into planner")
        return states

    def aoi(self, step: int) -> list[int]:
        """Age of Information: steps since each drone's latest delivered timestamp."""
        return [step - message.timestamp for message in self.planner_states(step)]

    def out_of_order_rate(self) -> float:
        """Fraction of delivered messages that arrived out of order."""
        if self._total_delivered_count == 0:
            return 0.0
        return self._out_of_order_count / self._total_delivered_count

    # ── internal ────────────────────────────────────────────────────────

    def _accept(self, messages: list, visible_step: int | None = None) -> None:
        for message in messages:
            drone_id = message.drone_id
            current = self._delivered.get(drone_id)
            self._total_delivered_count += 1
            # Detect out-of-order: a message whose send_timestamp is older than
            # the latest we have already accepted for this drone.
            if drone_id in self._latest_delivered_timestamp:
                if message.timestamp < self._latest_delivered_timestamp[drone_id]:
                    self._out_of_order_count += 1
            if current is None or message.timestamp >= current.timestamp:
                self._delivered[drone_id] = message
                self._latest_delivered_timestamp[drone_id] = message.timestamp
                if visible_step is not None:
                    self.visibility_records.append({
                        "drone_id": drone_id,
                        "send_step": message.timestamp,
                        "first_visible_step": int(visible_step),
                    })

    def communication_summary(self) -> dict[str, Any]:
        """Return aggregate communication statistics for this run."""
        actual_delays = [
            r["delay_steps"] for r in self.delivery_records
            if r["delay_steps"] >= 0
        ]
        dropped = sum(1 for r in self.delivery_records if r.get("dropped") == 1)
        total = len(self.delivery_records)
        return {
            "actual_mean_delay": float(np.mean(actual_delays)) if actual_delays else 0.0,
            "actual_p90_delay": float(np.percentile(actual_delays, 90)) if actual_delays else 0.0,
            "actual_loss_rate": dropped / total if total > 0 else 0.0,
            "actual_out_of_order_count": self._out_of_order_count,
            "actual_out_of_order_rate": self.out_of_order_rate(),
            "total_messages_sent": total,
            "total_messages_dropped": dropped,
        }
