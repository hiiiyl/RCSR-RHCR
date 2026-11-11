"""Continuous tracking diagnostic for the shared graph-planning interface.

Grid waypoint conflicts are filtered before physical integration. The PD
tracker is not a continuous collision-avoidance controller: physical contacts
are measured, never erased by changing a reached-node record afterwards.
"""
from __future__ import annotations

from math import hypot, sqrt
import numpy as np

from .continuous_dynamics import (
    ContinuousConfig, DoubleIntegrator, PDController, check_boundary,
    check_building_collision, check_uav_collision, continuous_to_grid,
    grid_to_continuous,
)
from .lifelong_city_env import LifelongCityEnv, LifelongConfig
from .motion_graph import legal_transition
from .rhcr_adapted import reference_projection

Point = tuple[int, int]
Float2 = tuple[float, float]


class ContinuousLifelongCityEnv(LifelongCityEnv):
    """Double-integrator tracking with sampled continuous safety diagnostics."""

    def __init__(self, config: LifelongConfig, dyn_config: ContinuousConfig | None = None):
        self.dyn_config = dyn_config or ContinuousConfig()
        super().__init__(config)

    @property
    def planning_goal_radius(self) -> float:
        """Keep the reached-node tolerance inside the physical service region."""
        radius = self.config.goal_radius - self.dyn_config.waypoint_tolerance / self.dyn_config.cell_size
        if radius < 0:
            raise ValueError("physical service radius must cover the waypoint tolerance")
        return radius

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        # Historical aliases count exposure samples, NOT contact events.
        self.continuous_collision_count = 0
        self.obstacle_collision_count = 0
        self.boundary_violation_count = 0
        self.uav_contact_events = 0
        self.obstacle_contact_events = 0
        self.boundary_violation_events = 0
        self.continuous_substeps = 0
        self._active_uav_contacts: set[tuple[int, int]] = set()
        self._active_obstacle_contacts: set[int] = set()
        self._active_boundary_violations: set[int] = set()
        self._active_discrete_contacts: set[tuple[int, int]] = set()
        self.minimum_continuous_pair_distance = float("inf")
        self._tracking_squared_error = 0.0
        self._tracking_sample_count = 0
        self.rejected_reached_node_updates = 0
        self._init_continuous_state()

    def _init_continuous_state(self) -> None:
        """Initialize from grid launch nodes; callable after tape hydration."""
        cfg = self.dyn_config
        self._continuous_positions = [list(grid_to_continuous(*p, cfg.cell_size)) for p in self.positions]
        self._continuous_velocities = [[0.0, 0.0] for _ in self.positions]
        self._integrators = [DoubleIntegrator(p, v, cfg) for p, v in zip(self._continuous_positions, self._continuous_velocities)]
        self.last_reached_nodes = list(self.positions)

    def _assert_state(self) -> None:
        """Validate records while permitting observed diagnostic collisions.

        Duplicate reached nodes remain visible instead of aborting a run or
        inventing a physically unexecuted separation manoeuvre.
        """
        for point in self.positions:
            if not self._valid(point):
                raise AssertionError("last reached node is not a free map cell")
            x, y = point
            if self.config.flight_altitude < self.building_heights[y, x] + self.config.vertical_clearance:
                raise AssertionError("last reached node violates height clearance")
        if np.any(self.energies < self.config.energy_reserve - 1e-9):
            raise AssertionError("UAV violated energy reserve")
        self.target_pool.assert_invariants(set(self.positions))

    def _target_reached(self, drone_idx: int, target) -> bool:
        """Evaluate service from physical position, not a stale graph record.

        The shared goal radius is expressed in grid cells and converted to
        metres here. Service is sampled at the end of a graph execution step;
        it does not imply that the physical state is collision-free.
        """
        x, y = self._integrators[drone_idx].pos
        tx, ty = grid_to_continuous(*target.position, self.dyn_config.cell_size)
        return hypot(x - tx, y - ty) <= self.config.goal_radius * self.dyn_config.cell_size

    def _record_contacts(self, positions: list[Float2]) -> None:
        """Count end-of-substep exposure and contiguous-contact onsets.

        A pair contributes one sample for each overlap; a UAV contributes one
        obstacle sample even when touching several cells. Events are sampled
        contact onsets, not independent accidents.
        """
        cfg = self.dyn_config
        pairs = set(check_uav_collision(positions, cfg.robot_radius))
        obstacles = {i for i, (x, y) in enumerate(positions) if check_building_collision(x, y, cfg.robot_radius, self.building_mask, cfg.cell_size)}
        boundary = {i for i, (x, y) in enumerate(positions) if not check_boundary(x, y, cfg.robot_radius, self.config.width, self.config.height, cfg.cell_size)}
        self.continuous_collision_count += len(pairs)
        self.obstacle_collision_count += len(obstacles)
        self.boundary_violation_count += len(boundary)
        self.uav_contact_events += len(pairs - self._active_uav_contacts)
        self.obstacle_contact_events += len(obstacles - self._active_obstacle_contacts)
        self.boundary_violation_events += len(boundary - self._active_boundary_violations)
        self._active_uav_contacts = pairs
        self._active_obstacle_contacts = obstacles
        self._active_boundary_violations = boundary
        self.continuous_substeps += 1
        distances = [hypot(a[0] - b[0], a[1] - b[1]) for i, a in enumerate(positions) for b in positions[i + 1:]]
        self.minimum_continuous_pair_distance = min(self.minimum_continuous_pair_distance, min(distances, default=float("inf")))

    def _update_reached_nodes(self) -> None:
        """Accept only a reached centre joined by a legal graph edge."""
        cfg = self.dyn_config
        for i, integrator in enumerate(self._integrators):
            position = integrator.pos
            candidate = continuous_to_grid(*position, cfg.cell_size)
            centre = grid_to_continuous(*candidate, cfg.cell_size)
            if hypot(position[0] - centre[0], position[1] - centre[1]) > cfg.waypoint_tolerance:
                continue
            if legal_transition(self.building_mask, self.last_reached_nodes[i], candidate):
                self.last_reached_nodes[i] = candidate
            elif candidate != self.last_reached_nodes[i]:
                self.rejected_reached_node_updates += 1

    def _execute_controlled_step(
        self, commanded_positions: list[Point], deviation_probability: float,
        tube_slice: list[set[Point]] | None = None,
        execution_context: dict | None = None,
        physical_audit_context: dict | None = None,
        emc_support_audit_context: dict | None = None,
        hold_mask: list[bool] | None = None,
    ) -> None:
        """Filter waypoints, integrate motion, then observe reached nodes.

        Explicit holds target the current continuous position; velocity is
        damped, not reset. Stochastic holds target the last reached grid centre.
        """
        cfg = self.dyn_config
        count = self.config.num_drones
        if len(commanded_positions) != count:
            raise ValueError("one reference is required for every UAV")
        holds = [False] * count if hold_mask is None else list(hold_mask)
        if len(holds) != count:
            raise ValueError("hold_mask must have one entry per UAV")
        old_nodes = list(self.positions)
        desired_nodes = []
        deviation_holds = []
        for i, (node, reference) in enumerate(zip(old_nodes, commanded_positions)):
            desired = node if holds[i] else reference_projection(self.building_mask, node, reference)
            deviation = self._execution_deviation(self.step_id, i, node, desired, deviation_probability)
            desired_nodes.append(node if deviation else desired)
            deviation_holds.append(bool(deviation))

        duplicate_start = len(set(old_nodes)) != count
        if duplicate_start:
            # Damping in place does not claim to repair a joint collision.
            resolved = list(old_nodes)
            holds = [True] * count
            interventions = {"start_occupied_hold_count": 0, "duplicate_destination_hold_count": 0}
        else:
            resolved = self._resolve_moves(desired_nodes)
            interventions = dict(self._last_safety_interventions)
        safety_holds = sum(a != b for a, b in zip(desired_nodes, resolved))
        waypoints = [self._integrators[i].pos if holds[i] else grid_to_continuous(*node, cfg.cell_size) for i, node in enumerate(resolved)]

        controller = PDController(cfg)
        for _ in range(cfg.substeps_per_planning_step):
            for i, integrator in enumerate(self._integrators):
                integrator.step(controller.control(integrator.pos, integrator.vel, waypoints[i]))
            positions = [integrator.pos for integrator in self._integrators]
            self._record_contacts(positions)
            self._tracking_squared_error += sum((p[0] - w[0]) ** 2 + (p[1] - w[1]) ** 2 for p, w in zip(positions, waypoints))
            self._tracking_sample_count += count

        self._continuous_positions = [list(integrator.pos) for integrator in self._integrators]
        self._continuous_velocities = [list(integrator.vel) for integrator in self._integrators]
        self._update_reached_nodes()
        # Do not resolve grid moves after integration: that would falsify the
        # observed progress without changing physical position or velocity.
        self._execute_reserved_step(list(self.last_reached_nodes))
        discrete_pairs = {(i, j) for i in range(count) for j in range(i + 1, count) if self.positions[i] == self.positions[j]}
        self.collision_count += len(discrete_pairs - self._active_discrete_contacts)
        self._active_discrete_contacts = discrete_pairs
        errors = [hypot(p[0] - w[0], p[1] - w[1]) for p, w in zip(self._continuous_positions, [grid_to_continuous(*node, cfg.cell_size) for node in commanded_positions])]
        self.step_records[-1].update({
            "collision_count": self.collision_count,
            "commanded_deviation_count": sum(deviation_holds),
            "explicit_hold_count": sum(holds),
            "duplicate_reached_node_hold": int(duplicate_start),
            "safety_conflict_hold_count": safety_holds,
            "start_occupied_hold_count": interventions.get("start_occupied_hold_count", 0),
            "duplicate_destination_hold_count": interventions.get("duplicate_destination_hold_count", 0),
            "mean_command_tracking_error": float(np.mean(errors)),
            "outside_nominal_occupancy_rate": float(np.mean([p != r for p, r in zip(self.positions, commanded_positions)])),
            "tube_coverage_rate": None, "mean_tube_size": None,
        })
        for i, integrator in enumerate(self._integrators):
            self.uav_action_records.append({
                "step": self.step_id, "drone_id": i,
                "continuous_x": integrator.pos[0], "continuous_y": integrator.pos[1],
                "velocity_x": integrator.vel[0], "velocity_y": integrator.vel[1],
                "last_reached_node": list(self.last_reached_nodes[i]),
                "applied_waypoint_x": waypoints[i][0], "applied_waypoint_y": waypoints[i][1],
                "command_tracking_error_m": errors[i],
                "deviation_hold": deviation_holds[i], "explicit_hold": holds[i],
            })

    def continuous_summary(self) -> dict:
        """Return separately labelled sampled-contact events and exposures."""
        cfg = self.dyn_config
        n = self.config.num_drones
        pair_samples = self.continuous_substeps * n * (n - 1) / 2
        uav_samples = self.continuous_substeps * n
        minimum = self.minimum_continuous_pair_distance
        return {
            # Historical aliases are retained for older analysis scripts.
            "continuous_collision_count": self.continuous_collision_count,
            "obstacle_collision_count": self.obstacle_collision_count,
            "boundary_violation_count": self.boundary_violation_count,
            "uav_contact_pair_substeps": self.continuous_collision_count,
            "obstacle_contact_uav_substeps": self.obstacle_collision_count,
            "boundary_violation_uav_substeps": self.boundary_violation_count,
            "uav_contact_events": self.uav_contact_events,
            "obstacle_contact_events": self.obstacle_contact_events,
            "boundary_violation_events": self.boundary_violation_events,
            "uav_contact_pair_seconds": cfg.control_dt * self.continuous_collision_count,
            "obstacle_contact_uav_seconds": cfg.control_dt * self.obstacle_collision_count,
            "boundary_violation_uav_seconds": cfg.control_dt * self.boundary_violation_count,
            "uav_contact_pair_substep_fraction": self.continuous_collision_count / max(pair_samples, 1),
            "obstacle_contact_uav_substep_fraction": self.obstacle_collision_count / max(uav_samples, 1),
            "boundary_violation_uav_substep_fraction": self.boundary_violation_count / max(uav_samples, 1),
            "minimum_pairwise_distance_m": None if not np.isfinite(minimum) else minimum,
            "applied_waypoint_tracking_rmse_m": sqrt(self._tracking_squared_error / max(self._tracking_sample_count, 1)),
            "continuous_substeps": self.continuous_substeps,
            "rejected_reached_node_updates": self.rejected_reached_node_updates,
        }
