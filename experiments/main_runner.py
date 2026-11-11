"""Run paired method comparisons on shared random tapes."""
from __future__ import annotations

import os
import sys
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
sys.dont_write_bytecode = True

import argparse
import dataclasses
import gzip
import json
import math
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "simulator"
sys.path.insert(0, str(PACKAGE / "src"))
sys.path.insert(0, str(PACKAGE / "analysis"))
from urban_lifelong_planning.revision_experiment import EpisodeSpec, run_episode
from metric_contract import normalize_case
import numpy as np

METRICS = ("p90_response_time", "mean_planning_start_error", "steady_state_throughput", "safety_interventions_per_uav_step")
METHODS = {
    "RCSR-RHCR": ("RCSR-Component-Task-History", "support_separated_forward_v1", "rcsr"),
    "RHCR-adapted": ("RHCR-adapted", "nominal", "rhcr"),
    "CC-RHCR": ("CC-RHCR", "nominal", "cc"),
    "Index-only EP": ("Index-only EP", "nominal", "index"),
    "Deterministic replay": ("Deterministic replay", "nominal", "deterministic"),
    "EMC-RHCR": ("EMC-RHCR", "nominal", "emc"),
    "Spatial replay no graph": ("Spatial replay no graph", "nominal", "spatial_no_graph"),
    "BaF-guided RHCR": ("BaF-guided RHCR", "nominal", "baf"),
}


def jsonable(value):
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value):
            return {k: jsonable(v) for k, v in value.items()}
        return {"__ordered_mapping__": [[jsonable(k), jsonable(v)] for k, v in value.items()]}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def run_one(job):
    display, seed, delay, output_subdir = job
    method, reference, slug = METHODS[display]
    path = ROOT / output_subdir / f"{slug}_s{seed}_d{delay}.json.gz"
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing.get("status") != "completed" or existing["display_method"] != display:
            raise ValueError(f"Unexpected existing result: {path}")
        return {"method": display, "seed": seed, "delay": delay, "status": "reused", "metrics": existing["metrics"]}
    # Match the recorded experiment settings, including task allocation and goal release.
    spec = EpisodeSpec(experiment="integrated_development_proposal", method=method,
        seed=seed, delay=delay, reference_policy=reference, goal_release=True)
    tape = PACKAGE / "tapes" / f"seed_{seed}" / "tape.json"
    started = perf_counter()
    summary, tasks, planning = run_episode(spec, tape)
    document = {"status": "completed", "display_method": display,
        "spec": dataclasses.asdict(spec), "summary": summary, "tasks": tasks, "planning": planning}
    normalized = normalize_case(document)
    document["metrics"] = {name: normalized["metrics"][name]["value"] for name in METRICS}
    document["python_version"] = platform.python_version()
    document["numpy_version"] = np.__version__
    document["runtime_note"] = "Performance execution; timings are excluded from the independent serial-cost comparison."
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(jsonable(document), stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    temporary.replace(path)
    return {"method": display, "seed": seed, "delay": delay, "status": "completed",
        "elapsed_s": round(perf_counter() - started, 2), "metrics": document["metrics"]}
