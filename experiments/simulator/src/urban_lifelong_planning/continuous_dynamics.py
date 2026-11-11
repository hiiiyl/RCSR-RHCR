"""2D double-integrator dynamics and PD waypoint tracking for Stage 3.

This module is deliberately separate from the discrete grid environment.
It provides the continuous execution layer that sits *below* the discrete
RHCR/RCSR planning interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isclose, isfinite
from typing import Tuple

import numpy as np

# ── types ──────────────────────────────────────────────────────────────────

Float2 = Tuple[float, float]


# ── configuration ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ContinuousConfig:
    """Continuous dynamics parameters, matching the reported experiment."""

    cell_size: float = 1.0
    control_dt: float = 0.05          # seconds per control substep
    planning_step_duration: float = 1.0  # seconds per discrete step
    substeps_per_planning_step: int = 20
    v_max: float = 1.8                # m/s
    a_max: float = 3.0                # m/s²
    robot_radius: float = 0.15        # m
    waypoint_tolerance: float = 0.30  # m — reached-node observation threshold
    Kp: float = 8.0                   # position gain
    Kd: float = 2.5                   # velocity damping

    def __post_init__(self) -> None:
        values = (self.cell_size, self.control_dt, self.planning_step_duration,
                  self.v_max, self.a_max, self.robot_radius,
                  self.waypoint_tolerance, self.Kp, self.Kd)
        if any(not isfinite(value) or value <= 0 for value in values):
            raise ValueError("continuous parameters must be finite and positive")
        if self.substeps_per_planning_step < 1 or int(self.substeps_per_planning_step) != self.substeps_per_planning_step:
            raise ValueError("substeps_per_planning_step must be a positive integer")
        if not isclose(self.control_dt * self.substeps_per_planning_step, self.planning_step_duration):
            raise ValueError("control substeps must cover exactly one discrete step")
        if self.waypoint_tolerance >= self.cell_size / 2:
            raise ValueError("waypoint tolerance must distinguish adjacent cell centres")


# ── grid ↔ continuous mapping ──────────────────────────────────────────────

def grid_to_continuous(grid_x: int, grid_y: int, cell_size: float = 1.0) -> Float2:
    """Map a discrete grid node to its cell center in continuous coordinates."""
    return ((grid_x + 0.5) * cell_size, (grid_y + 0.5) * cell_size)


def continuous_to_grid(x: float, y: float, cell_size: float = 1.0) -> tuple[int, int]:
    """Map continuous coordinates to their containing cell (no clamping)."""
    return (int(np.floor(x / cell_size)), int(np.floor(y / cell_size)))


# ── collision detection ────────────────────────────────────────────────────

def circle_rect_intersect(
    cx: float, cy: float, radius: float,
    rx: float, ry: float, rw: float, rh: float,
) -> bool:
    """Check if a circle intersects an axis-aligned rectangle."""
    closest_x = max(rx, min(cx, rx + rw))
    closest_y = max(ry, min(cy, ry + rh))
    dist_sq = (cx - closest_x) ** 2 + (cy - closest_y) ** 2
    return dist_sq < radius ** 2


def check_building_collision(
    x: float, y: float, radius: float,
    building_mask: np.ndarray, cell_size: float,
) -> bool:
    """Check if a circle at (x, y) intersects any building cell."""
    grid_x, grid_y = continuous_to_grid(x, y, cell_size)
    reach = max(1, ceil(radius / cell_size))
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            cx, cy = grid_x + dx, grid_y + dy
            if 0 <= cy < building_mask.shape[0] and 0 <= cx < building_mask.shape[1]:
                if building_mask[cy, cx]:
                    if circle_rect_intersect(
                        x, y, radius,
                        cx * cell_size, cy * cell_size, cell_size, cell_size,
                    ):
                        return True
    return False


def check_boundary(
    x: float, y: float, radius: float,
    width: int, height: int, cell_size: float,
) -> bool:
    """Check if a circle is within the map boundary."""
    map_w, map_h = width * cell_size, height * cell_size
    return (radius <= x <= map_w - radius) and (radius <= y <= map_h - radius)


def check_uav_collision(
    positions: list[Float2], radius: float,
) -> list[tuple[int, int]]:
    """Return pairs of UAV indices whose circles intersect."""
    pairs = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            if dx * dx + dy * dy < (2 * radius) ** 2:
                pairs.append((i, j))
    return pairs


# ── PD controller ──────────────────────────────────────────────────────────

class PDController:
    """Saturated PD waypoint tracker shared by all methods."""

    def __init__(self, config: ContinuousConfig):
        self.Kp = config.Kp
        self.Kd = config.Kd
        self.a_max = config.a_max
        self.v_max = config.v_max

    def control(
        self,
        position: Float2,
        velocity: Float2,
        waypoint: Float2,
    ) -> Float2:
        """Compute acceleration command."""
        px, py = position
        vx, vy = velocity
        wx, wy = waypoint
        ax = self.Kp * (wx - px) - self.Kd * vx
        ay = self.Kp * (wy - py) - self.Kd * vy
        # Clip acceleration magnitude
        norm = np.sqrt(ax * ax + ay * ay)
        if norm > self.a_max:
            scale = self.a_max / norm
            ax *= scale
            ay *= scale
        return (float(ax), float(ay))


# ── double integrator ──────────────────────────────────────────────────────

class DoubleIntegrator:
    """2D double-integrator dynamics for one UAV."""

    def __init__(self, position: Float2, velocity: Float2, config: ContinuousConfig):
        self.position = list(position)
        self.velocity = list(velocity)
        self.config = config

    def step(self, acceleration: Float2) -> None:
        """Semi-implicit Euler: update/cap velocity, then update position."""
        dt = self.config.control_dt
        self.velocity[0] += acceleration[0] * dt
        self.velocity[1] += acceleration[1] * dt
        # Clip velocity
        speed = np.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)
        if speed > self.config.v_max:
            scale = self.config.v_max / speed
            self.velocity[0] *= scale
            self.velocity[1] *= scale
        self.position[0] += self.velocity[0] * dt
        self.position[1] += self.velocity[1] * dt

    @property
    def pos(self) -> Float2:
        return (self.position[0], self.position[1])

    @property
    def vel(self) -> Float2:
        return (self.velocity[0], self.velocity[1])
