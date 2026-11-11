"""Window, fleet, communication, continuous-execution and timing experiments."""
from __future__ import annotations

import os
import sys
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
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
METRICS = ("p90_response_time", "mean_planning_start_error", "steady_state_throughput", "safety_interventions_per_uav_step")
METHODS = {
    "RCSR-RHCR": ("RCSR-Component-Task-History", "support_separated_forward_v1", "rcsr"),
    "RHCR-adapted": ("RHCR-adapted", "nominal", "rhcr"),
    "CC-RHCR": ("CC-RHCR", "nominal", "cc"),
}


def jsonable(value):
    import numpy as np
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {key: jsonable(item) for key, item in value.items()}
        return {"__ordered_mapping__": [[jsonable(key), jsonable(item)] for key, item in value.items()]}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def tape_path(seed, fleet, steps=1000):
    if fleet == 6 and steps == 1000:
        return PACKAGE / "tapes" / f"seed_{seed}" / "tape.json"
    return ROOT / "tapes" / f"n{fleet}_t{steps}" / f"seed_{seed}" / "tape.json"


def prepare_tapes(jobs):
    sys.path.insert(0, str(PACKAGE / "src"))
    from urban_lifelong_planning.formal_random_tape import build_tape, write_tape
    from urban_lifelong_planning.lifelong_city_env import LifelongConfig
    for seed, fleet, steps in sorted({(job["seed"], job["fleet"], job["steps"]) for job in jobs}):
        path = tape_path(seed, fleet, steps)
        if not path.exists():
            config = LifelongConfig(seed=seed, episode_steps=steps, num_drones=fleet,
                active_targets=2*fleet, initial_energy=1500.)
            write_tape(path, build_tape(seed, config))


def result_path(job):
    slug = METHODS[job["display_method"]][2]
    return ROOT / "episodes" / job["suite"] / (
        f"{slug}_s{job['seed']}_d{job['delay']}_n{job['fleet']}_k{job['commit_window']}"
        f"_{job['condition']}_t{job['steps']}.json.gz")


def run_one(job):
    path = result_path(job)
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            result = json.load(stream)
        if result["status"] != "completed" or result["experiment_job"] != job:
            raise ValueError(f"Unexpected existing result: {path}")
        return {**job, "status": "reused", "metrics": result["metrics"]}
    sys.path.insert(0, str(PACKAGE / "src"))
    sys.path.insert(0, str(PACKAGE / "analysis"))
    from urban_lifelong_planning.revision_experiment import EpisodeSpec, run_episode
    from metric_contract import normalize_case
    import numpy as np
    display = job["display_method"]
    method, reference, _ = METHODS[display]
    spec = EpisodeSpec(experiment="integrated_development_proposal", method=method,
        seed=job["seed"], delay=job["delay"], fleet=job["fleet"], steps=job["steps"],
        commit_window=job["commit_window"], reference_policy=reference, goal_release=True)
    if job["suite"] in ("async", "continuous"):
        sys.path.insert(0, str(ROOT))
        from extension.run_extension import run_extension_episode
        runner = lambda: run_extension_episode(spec, tape_path(spec.seed, spec.fleet, spec.steps),
            condition=job["condition"], continuous=job["suite"] == "continuous")
    else:
        runner = lambda: run_episode(spec, tape_path(spec.seed, spec.fleet, spec.steps))
    started = perf_counter()
    summary, tasks, planning = runner()
    document = {"status": "completed", "display_method": display, "experiment_job": job,
        "spec": dataclasses.asdict(spec), "summary": summary, "tasks": tasks, "planning": planning}
    normalized = normalize_case(document)
    document["metrics"] = {name: normalized["metrics"][name]["value"] for name in METRICS}
    document["python_version"] = platform.python_version()
    document["numpy_version"] = np.__version__
    document["runtime_note"] = ("Fresh worker, serial episodes, numerical-library threads=1."
        if job["suite"] == "cost" else "Parallel experiment; wall-clock timing is not used for computational-cost comparisons.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(jsonable(document), stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    temporary.replace(path)
    return {**job, "status": "completed", "elapsed_s": round(perf_counter()-started, 2), "metrics": document["metrics"]}


def make_jobs(args):
    jobs = []
    def add(method, seed, delay=4, fleet=6, window=2, condition="fixed"):
        jobs.append(dict(suite=args.suite, display_method=method, seed=seed, delay=delay,
            fleet=fleet, commit_window=window, condition=condition, steps=args.steps))
    seeds = args.seeds or list(range(9400, 9420))
    if args.suite == "window":
        # k=2 and RCSR's matched runs already exist in the recorded main panel.
        for seed in seeds:
            for delay in (4, 6):
                for window in (1, 4, 6, 8):
                    add("CC-RHCR", seed, delay=delay, window=window)
    elif args.suite in ("scale", "cost"):
        for seed in seeds:
            for fleet in (args.fleets or ([6, 12, 18] if args.suite == "cost" else [12, 18])):
                for method in (args.methods or list(METHODS)):
                    add(method, seed, fleet=fleet)
    elif args.suite in ("async", "continuous"):
        conditions = args.conditions or (["moderate", "severe"] if args.suite == "async" else ["fixed", "moderate"])
        for seed in seeds:
            for condition in conditions:
                for method in (args.methods or list(METHODS)):
                    add(method, seed, delay=0, condition=condition)
    return jobs
