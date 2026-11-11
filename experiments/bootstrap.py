"""Seed-level means and paired percentile bootstrap intervals."""
from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import math
import random
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SEEDS = tuple(range(9400, 9420))
DELAYS = (2, 4, 6)
BOOTSTRAP_SEED = 20261107
BOOTSTRAP_REPLICATES = 10000
EXISTING = {
    "RCSR-RHCR": "configuration_19",
    "RHCR-adapted": "configuration_20",
    "CC-RHCR": "configuration_21",
    "Index-only EP": "configuration_23",
    "Deterministic replay": "configuration_24",
    "EMC-RHCR": "configuration_00",
}
NEW_METHODS = ("Spatial replay no graph", "BaF-guided RHCR")
METHODS = tuple(EXISTING) + NEW_METHODS
# The scale is applied only when exporting; calculations retain the input units.
METRICS = {
    "p90_response_time": ("p90_steps", "P90 响应（步）↓", 1.0, 2),
    "mean_planning_start_error": ("start_error_grid", "修复后起点误差（格）↓", 1.0, 4),
    "steady_state_throughput": ("throughput_tasks_per_step", "吞吐量（任务/步）↑", 1.0, 4),
    "safety_interventions_per_uav_step": (
        "interventions_per_1000_uav_steps", "干预（次/千 UAV-step）↓", 1000.0, 3
    ),
}


def finite_value(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected a numeric metric or null, received {value!r}")
    return float(value) if math.isfinite(value) else None


def quantile(values, probability):
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    low = math.floor(location)
    high = math.ceil(location)
    return ordered[low] + (ordered[high] - ordered[low]) * (location - low)


def summarize(seed_values, paired=False):
    values = [seed_values[seed] for seed in SEEDS if seed_values[seed] is not None]
    result = {
        "mean": statistics.mean(values) if values else None,
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "n_valid": len(values),
        "n_expected": len(SEEDS),
        "ci95": [None, None],
    }
    if paired and len(values) > 1:
        # Same random stream, mean and percentile convention as the supplied analysis.
        rng = random.Random(BOOTSTRAP_SEED)
        draws = [statistics.mean(rng.choices(values, k=len(values)))
                 for _ in range(BOOTSTRAP_REPLICATES)]
        result["ci95"] = [quantile(draws, 0.025), quantile(draws, 0.975)]
    return result


def read_seed_values(row):
    values = {int(item["seed"]): finite_value(item["value"])
              for item in row["seed_rows"]}
    if len(row["seed_rows"]) != len(SEEDS) or set(values) != set(SEEDS):
        raise ValueError("Supplied result does not contain the expected twenty seed rows")
    return values
