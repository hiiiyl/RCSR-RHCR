"""Tape-backed environment adapter for formal infrastructure checks.

The adapter changes only the source of exogenous randomness and telemetry
delivery times.  All inherited controller, allocator, target-pool and safety
semantics remain the frozen implementations.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from .delayed_state_channel import DelayedStateChannel
from .formal_random_tape import load_tape
from .lifelong_city_env import LifelongCityEnv, LifelongConfig
from .target_pool import PublicTargetPool, TargetRecord

Point = tuple[int, int]


class UnexpectedRNGGuard:
    """Counts and rejects accidental post-launch environment RNG use."""

    def __init__(self) -> None:
        self.call_count = 0

    def _unexpected(self, *_args, **_kwargs):
        self.call_count += 1
        raise AssertionError("formal tape run attempted an unexpected environment RNG call")

    random = _unexpected
    integers = _unexpected
    uniform = _unexpected


class FormalTapeCityEnv(LifelongCityEnv):
    """A `LifelongCityEnv` hydrated from one immutable explicit-index tape."""

    def __init__(self, config: LifelongConfig, tape_path: Path):
        self.tape, self.tape_sha256 = load_tape(tape_path)
        metadata = self.tape["metadata"]
        if metadata["seed"] != config.seed:
            raise ValueError("formal tape seed does not match LifelongConfig.seed")
        if metadata["episode_steps"] != config.episode_steps or metadata["uav_count"] != config.num_drones:
            raise ValueError("formal tape dimensions do not match LifelongConfig")
        self._hydrating_base_initial_state = True
        super().__init__(config)
        self._hydrating_base_initial_state = False
        self._hydrate_initial_state()
        self._rng_guard = UnexpectedRNGGuard()
        self.rng = self._rng_guard
        self.target_pool.rng = self._rng_guard
        self._replacement_index = 0
        self.deviation_tape_records: list[dict[str, int]] = []
        self.target_tape_records: list[dict[str, object]] = []
        self.delay_channels: list[DelayedStateChannel] = []

    @property
    def unexpected_rng_call_count(self) -> int:
        return self._rng_guard.call_count

    def _hydrate_initial_state(self) -> None:
        state = self.tape["initial_state"]
        self.building_mask = np.asarray(state["building_mask"], dtype=bool)
        self.building_heights = np.asarray(state["building_heights"], dtype=np.float32)
        self.positions = [tuple(point) for point in state["uav_positions"]]
        self.trajectories = [[point] for point in self.positions]
        self.energies = np.asarray(state["energies"], dtype=np.float64)
        pool = PublicTargetPool(self.config.active_targets, self.rng, allow_coincident_targets=True)
        pool.active = {
            int(item["target_id"]): TargetRecord(int(item["target_id"]), tuple(item["position"]), int(item["release_step"]))
            for item in state["initial_targets"]
        }
        pool._next_id = max(pool.active, default=-1) + 1
        self.target_pool = pool
        self._assert_state()

    def _sample_free_cell(self, excluded: set[Point]) -> Point:
        if self._hydrating_base_initial_state:
            return super()._sample_free_cell(excluded)
        sequence = self.tape["target_replacement_sequence"]
        if self._replacement_index >= len(sequence):
            raise RuntimeError("target replacement tape exhausted")
        point = tuple(sequence[self._replacement_index])
        occupied_uav_ids = [index for index, position in enumerate(self.positions) if position == point]
        active_target_ids = [target_id for target_id, target in self.target_pool.active.items() if target.position == point]
        self.target_tape_records.append({
            "step": self.step_id,
            "replacement_index": self._replacement_index,
            "position": point,
            "spawned_on_occupied_uav": int(bool(occupied_uav_ids)),
            "occupied_uav_ids": occupied_uav_ids,
            "spawned_on_active_target": int(bool(active_target_ids)),
            "active_target_ids": active_target_ids,
        })
        self._replacement_index += 1
        # A fixed public-target tape must preserve one replacement coordinate
        # per index across all methods. UAV occupancy is method-dependent, so
        # rejecting a coordinate merely because a UAV currently occupies it
        # would make a valid shared tape abort. The target pool itself permits
        # public targets at occupied cells; completion still requires a real
        # assignment and the existing true-position/service check. An active
        # coordinate may also repeat: it represents a distinct public task
        # with a separate target ID. Only map-invalid coordinates are errors.
        if not self._valid(point):
            raise AssertionError("explicit target tape point is not a free map cell")
        return point

    def _execution_deviation(self, step: int, drone_idx: int, current: Point, desired: Point, deviation_probability: float) -> bool:
        metadata = self.tape["metadata"]
        if deviation_probability != metadata["p_deviation"]:
            raise AssertionError("formal run changed frozen deviation probability")
        value = int(self.tape["execution_deviation"][step][drone_idx])
        self.deviation_tape_records.append({"step": step, "drone_id": drone_idx, "deviation": value})
        return desired != current and bool(value)

    def _make_delayed_state_channel(self, delay_steps: int) -> DelayedStateChannel:
        schedule = self.tape["message_delay"].get(str(delay_steps))
        if schedule is None:
            raise ValueError(f"tape has no explicit delay schedule for {delay_steps}")

        def provider(send_step: int, drone_id: int) -> int:
            return int(schedule[send_step][drone_id])

        channel = DelayedStateChannel(self.config.num_drones, delay_steps, provider)
        self.delay_channels.append(channel)
        return channel

    def tape_audit(self) -> dict[str, Any]:
        return {
            "tape_sha256": self.tape_sha256,
            "unexpected_rng_call_count": self.unexpected_rng_call_count,
            "execution_deviation_entries_consumed": len(self.deviation_tape_records),
            "target_replacement_entries_consumed": len(self.target_tape_records),
            "target_replacement_sequence_length": len(self.tape["target_replacement_sequence"]),
            "metadata": self.tape["metadata"],
        }
