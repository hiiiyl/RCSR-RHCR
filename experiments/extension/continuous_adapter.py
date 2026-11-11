"""PD execution adapter for the unchanged V3 graph inference kernels.

The common-timestamp channel admits only delivered, graph-valid joint anchors.
The physical environment retains all confirmed observations, including repeated
nodes; admission never edits those observations or the physical state. Service
history and reference control continue to use the discrete graph surrogate.
"""
from __future__ import annotations

from math import hypot
from numbers import Integral

import numpy as np

from urban_lifelong_planning.continuous_dynamics import grid_to_continuous
from urban_lifelong_planning.motion_graph import legal_cell
from urban_lifelong_planning.revision_experiment import TapeContinuousEnv


def make_graph_snapshot_validator(buildings):
    """Return a pure admission predicate for publicly delivered snapshots.

    Install this predicate in the common-timestamp channel for every continuous
    method. The channel keeps its previous accepted snapshot when this returns
    False. Its admission and age diagnostics belong to the channel, not to this
    predicate. No live environment, task outcome, or position enters it.
    """
    graph = np.asarray(buildings, dtype=bool).copy()
    if graph.ndim != 2 or not all(graph.shape):
        raise ValueError("snapshot validator requires a nonempty 2D motion graph")
    graph.flags.writeable = False

    def admissible(messages):
        if not messages:
            return False
        if {message.drone_id for message in messages} != set(range(len(messages))):
            return False
        if len({message.timestamp for message in messages}) != 1:
            return False
        positions = []
        for message in messages:
            point = message.position
            if (not isinstance(point, (tuple, list)) or len(point) != 2
                    or any(not isinstance(value, Integral) for value in point)):
                return False
            point = tuple(point)
            if not legal_cell(graph, point):
                return False
            positions.append(point)
        return len(set(positions)) == len(positions)

    return admissible


class ContinuousSurrogateTapeEnv(TapeContinuousEnv):
    """Original PD environment plus downstream physical error summaries.

    It can replace the runner's FormalTapeCityEnv constructor while the runner
    calls its synchronized graph inference path. The runner must label the
    extension as continuous externally and retain the surrogate scope fields.
    """

    def _execute_controlled_step(self, *args, **kwargs):
        # A complete decision record exists before execution. Record truth only
        # after that decision, once per planning cycle, never as inference input.
        if self.planning_records:
            record = self.planning_records[-1]
            if record["step"] == self.step_id:
                starts = record["control_starts"]
                if len(starts) != self.config.num_drones:
                    raise ValueError("one selected control start is required per UAV")
                positions = [integrator.pos for integrator in self._integrators]
                centres = [grid_to_continuous(*point, self.dyn_config.cell_size)
                           for point in starts]
                record["mean_continuous_start_error_m"] = float(np.mean([
                    hypot(p[0] - centre[0], p[1] - centre[1])
                    for p, centre in zip(positions, centres)]))
                record["physical_positions_before_execution_m"] = [list(p) for p in positions]
                record["physical_velocities_before_execution_m_per_s"] = [
                    list(integrator.vel) for integrator in self._integrators]
                record["graph_observations_before_execution"] = [list(p) for p in self.positions]
        return super()._execute_controlled_step(*args, **kwargs)

    def summary(self):
        result = super().summary()
        result.update(self.continuous_summary())
        errors = [record["mean_continuous_start_error_m"]
                  for record in self.planning_records
                  if "mean_continuous_start_error_m" in record]
        if errors:
            result["mean_continuous_start_error_m"] = float(np.mean(errors))
            result["p95_mean_continuous_start_error_m"] = float(np.percentile(errors, 95))
        result.update({
            "execution_model": "planar_double_integrator_PD",
            "inference_state_semantics": "discrete_graph_surrogate",
            "conditional_posterior_valid_scope": "discrete_graph_surrogate_only",
            "physical_posterior_available": False,
            "reference_tracker_compatibility_scope": "graph_surrogate_kernel",
            "true_support_coverage_semantics": "confirmed_graph_node_membership",
            "mean_planning_start_error_semantics": "confirmed_graph_node_error_in_cells",
            "continuous_service_semantics": "physical_centre_distance_at_graph_step_end",
            "continuous_interface_version": "service_margin_v2",
            "planning_goal_radius_cells": self.planning_goal_radius,
            "physical_service_radius_m": self.config.goal_radius * self.dyn_config.cell_size,
            "terminal_tracking_margin_m": self.dyn_config.waypoint_tolerance,
            "continuous_contact_semantics": "sampled_control_endpoint_onsets_and_exposure",
            "duplicate_confirmed_node_steps": sum(
                record.get("duplicate_reached_node_hold", 0) for record in self.step_records),
            "continuous_hold_semantics": "current_position_waypoint_with_PD_velocity_damping",
        })
        return result


def make_continuous_env_factory(captured):
    """Capture created environments for downstream reporting by the runner."""
    def factory(config, tape_path):
        env = ContinuousSurrogateTapeEnv(config, tape_path)
        captured.append(env)
        return env
    return factory
