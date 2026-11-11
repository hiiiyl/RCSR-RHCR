"""Immutable, explicitly indexed exogenous randomness for formal episodes.

This module is experiment infrastructure.  It does not implement a planner,
allocator, executor, target policy, or evaluation metric.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from .lifelong_city_env import LifelongCityEnv, LifelongConfig


TAPE_VERSION = 1
FORMAL_DELAYS = (0, 2, 4, 6)


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def payload_sha256(payload: dict[str, Any]) -> str:
    return sha256(canonical_bytes(payload)).hexdigest()


def _initial_state(env: LifelongCityEnv) -> dict[str, Any]:
    return {
        "building_mask": env.building_mask.astype(int).tolist(),
        "building_heights": env.building_heights.astype(float).tolist(),
        "uav_positions": [list(point) for point in env.positions],
        "energies": [float(value) for value in env.energies],
        "initial_targets": [
            {"target_id": record.target_id, "position": list(record.position), "release_step": record.release_step}
            for _, record in sorted(env.target_pool.active.items())
        ],
    }


def build_tape(seed: int, config: LifelongConfig) -> dict[str, Any]:
    """Create a tape without running a planning or execution episode."""
    env = LifelongCityEnv(config)
    if config.seed != seed:
        raise ValueError("config.seed must equal the tape seed")
    initial = _initial_state(env)
    free_cells = [
        (x, y)
        for y in range(config.height)
        for x in range(config.width)
        if not bool(env.building_mask[y, x])
    ]
    if not free_cells:
        raise RuntimeError("formal tape map has no free cells")
    deviation_rng = np.random.default_rng(np.random.SeedSequence([seed, 0xD3E1A7]))
    target_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x7A9E7]))
    # An episode cannot consume more replacements than completions; the upper
    # bound is one completion per UAV and step.  Each entry is a fixed point,
    # never a method-dependent RNG draw.
    replacement_count = config.episode_steps * config.num_drones
    replacements = [list(free_cells[int(target_rng.integers(0, len(free_cells)))]) for _ in range(replacement_count)]
    deviations = (deviation_rng.random((config.episode_steps, config.num_drones)) < 0.25).astype(int).tolist()
    message_delay = {
        str(delay): [[delay for _ in range(config.num_drones)] for _ in range(config.episode_steps + 1)]
        for delay in FORMAL_DELAYS
    }
    payload: dict[str, Any] = {
        "metadata": {
            "tape_version": TAPE_VERSION,
            "seed": int(seed),
            "episode_steps": int(config.episode_steps),
            "uav_count": int(config.num_drones),
            "active_targets": int(config.active_targets),
            "p_deviation": 0.25,
            "delay_labels": list(FORMAL_DELAYS),
            "generator": "explicit-indexed-formal-tape-v1",
            "planning_or_execution_run": False,
        },
        "initial_state": initial,
        "execution_deviation": deviations,
        "message_delay": message_delay,
        "target_replacement_sequence": replacements,
    }
    payload["metadata"]["payload_sha256"] = payload_sha256(payload)
    return payload


def write_tape(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(payload) + b"\n")
    digest = sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return digest


def load_tape(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    payload = json.loads(raw)
    claimed = payload["metadata"].get("payload_sha256")
    copy = json.loads(json.dumps(payload))
    copy["metadata"].pop("payload_sha256", None)
    if claimed != payload_sha256(copy):
        raise ValueError(f"tape content hash mismatch: {path}")
    return payload, digest


def tape_manifest(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("seed_*/tape.json")):
        payload, digest = load_tape(path)
        rows.append({
            "seed": payload["metadata"]["seed"],
            "path": path.relative_to(root).as_posix(),
            "sha256": digest,
            "episode_steps": payload["metadata"]["episode_steps"],
            "uav_count": payload["metadata"]["uav_count"],
        })
    return rows


def frozen_config_dict(config: LifelongConfig) -> dict[str, Any]:
    return asdict(config)
