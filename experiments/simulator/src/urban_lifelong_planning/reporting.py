"""Stage-1 CSV logging and map/trajectory visualization."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .lifelong_city_env import LifelongCityEnv


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(env: LifelongCityEnv, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(env.config), indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(output_dir / "episode_summaries.csv", [env.summary()])
    write_csv(output_dir / "step_records.csv", env.step_records)
    write_csv(output_dir / "planning_records.csv", env.planning_records)
    write_csv(output_dir / "target_records.csv", [record.as_dict() for record in env.target_pool.completed])
    _scenario_figure(env, output_dir / "scenario.png")
    _trajectory_figure(env, output_dir / "trajectories.png")
    if env.last_tubes:
        _tube_figure(env, output_dir / "aoi_tubes.png")


def _base_axes(env: LifelongCityEnv):
    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    axis.imshow(env.building_mask, origin="lower", cmap="Greys", interpolation="nearest")
    axis.set(xlim=(-0.5, env.config.width - 0.5), ylim=(-0.5, env.config.height - 0.5), xlabel="x", ylabel="y")
    return figure, axis


def _scenario_figure(env: LifelongCityEnv, path: Path) -> None:
    figure, axis = _base_axes(env)
    targets = np.asarray([target.position for target in env.target_pool.active.values()])
    axis.scatter(targets[:, 0], targets[:, 1], marker="*", s=65, color="#d62728", label="active public target")
    positions = np.asarray(env.positions)
    axis.scatter(positions[:, 0], positions[:, 1], marker="o", s=45, color="#1f77b4", label="UAV")
    axis.set_title(f"Stage 1 final scenario (step {env.step_id})")
    axis.legend(loc="upper right")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _trajectory_figure(env: LifelongCityEnv, path: Path) -> None:
    figure, axis = _base_axes(env)
    colors = plt.cm.tab10.colors
    for drone_idx, trajectory in enumerate(env.trajectories):
        points = np.asarray(trajectory)
        axis.plot(points[:, 0], points[:, 1], linewidth=1.3, color=colors[drone_idx % len(colors)], label=f"UAV {drone_idx}")
        axis.scatter(points[0, 0], points[0, 1], s=18, color=colors[drone_idx % len(colors)])
    axis.set_title("Stage 1 complete UAV trajectories")
    axis.legend(loc="upper right", ncol=2, fontsize=8)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _tube_figure(env: LifelongCityEnv, path: Path) -> None:
    figure, axis = _base_axes(env)
    colors = plt.cm.tab10.colors
    for drone_idx, tube in enumerate(env.last_tubes):
        cells = np.asarray(sorted(set().union(*tube)))
        if len(cells):
            axis.scatter(cells[:, 0], cells[:, 1], s=9, alpha=0.23, color=colors[drone_idx % len(colors)], label=f"UAV {drone_idx} tube")
    axis.set_title("AoI-expanded future occupancy tubes (last planning cycle)")
    axis.legend(loc="upper right", ncol=2, fontsize=7)
    figure.savefig(path, dpi=160)
    plt.close(figure)
