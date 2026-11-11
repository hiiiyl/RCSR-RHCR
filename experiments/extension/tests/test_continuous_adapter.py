"""Focused continuous-interface checks; no formal experiment matrix."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

EXTENSION = Path(__file__).resolve().parents[1]
SOURCE = EXTENSION.parent / "simulator" / "src"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(EXTENSION))

from continuous_adapter import ContinuousSurrogateTapeEnv, make_graph_snapshot_validator
from urban_lifelong_planning.continuous_env import ContinuousLifelongCityEnv
from urban_lifelong_planning.delayed_state_channel import StateMessage
from urban_lifelong_planning.lifelong_city_env import LifelongConfig
from urban_lifelong_planning.rhcr_adapted import IssuedPathAction
from urban_lifelong_planning.rhcr_adapted import RHCRAdapted, RHCRConfig
from urban_lifelong_planning.task_endpoint_conditioning import PublicTaskNotice
from urban_lifelong_planning.task_history import TaskHistoryComponentEstimator, history_for_control


def message(i, point, timestamp=0):
    return StateMessage(i, timestamp, point, 100., None, 0, -1)


def environment():
    config = LifelongConfig(width=12, height=12, num_drones=2,
                            active_targets=4, num_buildings=0, episode_steps=10)
    # Build the ordinary continuous state without a tape. The adapter adds only
    # downstream diagnostics; tape initialization is covered by the runner.
    env = ContinuousSurrogateTapeEnv.__new__(ContinuousSurrogateTapeEnv)
    env._hydrating_base_initial_state = True
    ContinuousLifelongCityEnv.__init__(env, config)
    env._execution_deviation = lambda *args: False
    env._sample_free_cell = lambda excluded: (10, 10)
    return env


def test_admission_uses_only_unchanged_delivered_joint_snapshot():
    buildings = np.zeros((5, 5), dtype=bool)
    buildings[3, 3] = True
    accept = make_graph_snapshot_validator(buildings)
    frames = [message(0, (1, 1), 2), message(1, (2, 1), 2)]
    before = deepcopy(frames)
    assert accept(frames)
    assert not accept([frames[0], replace(frames[1], position=(1, 1))])
    assert not accept([frames[0], replace(frames[1], position=(3, 3))])
    assert not accept([frames[0], replace(frames[1], timestamp=1)])
    assert not accept([frames[0], replace(frames[1], position=(-1, 1))])
    assert frames == before
    buildings[:] = False
    assert not accept([frames[0], replace(frames[1], position=(3, 3))])


def test_hold_damps_velocity_without_teleport_or_instant_stop():
    env = environment()
    env._integrators[0].velocity = [1., 0.]
    start = env._integrators[0].pos
    env.planning_records.append({"step": 0, "control_starts": list(env.positions), "planning_time_ms": 0.})
    env._execute_controlled_step([(10, 10), (0, 0)], 0., hold_mask=[True, True])
    first_record = env.planning_records[0]
    assert first_record["physical_positions_before_execution_m"][0] == list(start)
    assert first_record["physical_velocities_before_execution_m_per_s"][0] == [1., 0.]
    assert env._integrators[0].pos != start
    assert env._integrators[0].vel != (0., 0.)
    assert abs(env._integrators[0].pos[0] - start[0]) < 1.
    assert env.uav_action_records[0]["applied_waypoint_x"] == start[0]
    assert env.summary()["physical_posterior_available"] is False


def test_duplicate_confirmations_remain_observed_and_contacts_are_counted():
    env = environment()
    env.positions = [(3, 3), (3, 3)]
    env._init_continuous_state()
    env._execute_controlled_step([(4, 3), (2, 3)], 0.)
    assert env.positions == [(3, 3), (3, 3)]
    assert all(integrator.pos == (3.5, 3.5) for integrator in env._integrators)
    summary = env.summary()
    assert summary["duplicate_confirmed_node_steps"] == 1
    assert summary["uav_contact_events"] == 1
    assert summary["uav_contact_pair_substeps"] == 20
    assert summary["uav_contact_pair_substep_fraction"] == 1.
    env._execute_controlled_step([(4, 3), (2, 3)], 0.)
    assert env.continuous_summary()["uav_contact_events"] == 1
    assert env.continuous_summary()["uav_contact_pair_substeps"] == 40


def test_physical_service_can_disagree_with_graph_surrogate():
    env = environment()
    target = SimpleNamespace(position=(3, 3))
    env.positions[0] = (3, 3)
    env._integrators[0].position = [6.5, 3.5]
    assert not env._target_reached(0, target)
    env.positions[0] = (0, 0)
    env._integrators[0].position = [3.5, 3.5]
    assert env._target_reached(0, target)


def test_terminal_margin_rejects_premature_boundary_stop():
    env = environment()
    target = SimpleNamespace(position=(3, 3))
    env.positions[0] = (2, 3)
    env._integrators[0].position = [2.48, 3.5]
    assert not env._target_reached(0, target)
    graph = np.zeros((12, 12), dtype=bool)
    old = RHCRAdapted(RHCRConfig()).plan(graph, [(2, 3)], [(3, 3)])[0]
    assert all(p == (2, 3) for p in old)
    planner = RHCRAdapted(RHCRConfig(), goal_radius=env.planning_goal_radius)
    corrected = planner.plan(graph, [(2, 3)], [(3, 3)])[0]
    assert corrected[1] == (3, 3)
    # The full confirmation neighbourhood of the terminal node is serviceable.
    for dx, dy in ((.30, 0), (-.30, 0), (0, .30), (0, -.30)):
        env._integrators[0].position = [3.5 + dx, 3.5 + dy]
        assert env._target_reached(0, target)


def test_terminal_margin_scales_in_metres():
    env = environment()
    env.dyn_config = replace(env.dyn_config, cell_size=2.)
    assert env.planning_goal_radius == pytest.approx(.85)


def test_physical_history_mismatch_keeps_same_cycle_prior_fallback():
    estimator = TaskHistoryComponentEstimator(np.zeros((6, 6), bool), .25, 256)
    messages = [message(0, (0, 0))]
    paths = [{0: [(0, 0)]}]
    actions = [[IssuedPathAction(1, 0, 0, True)]]
    prior = estimator.estimate(messages, paths, actions, 1)
    completed = PublicTaskNotice(0, (3, 3), 0, 0, 1)
    control, audit = history_for_control(estimator, prior, messages, paths, actions, 1,
                                        completed=(completed,), service_radius=1.)
    assert control == prior
    assert audit["conditional_posterior_valid"] is False
    assert audit["conditioning_fallback_count"] == 1
    assert audit["conditional_evidence_mass"] is None
    assert audit["history_control_source"] == "unconditioned_component_whole_cycle"


def test_graph_observations_keep_legality_and_actual_tolerance():
    env = environment()
    env.last_reached_nodes[0] = (2, 2)
    env._integrators[0].position = [3.95, 2.5]
    env._update_reached_nodes()
    assert env.last_reached_nodes[0] == (2, 2)
    env._integrators[0].position = [3.5, 2.5]
    env._update_reached_nodes()
    assert env.last_reached_nodes[0] == (3, 2)
    env._integrators[0].position = [5.5, 2.5]
    env._update_reached_nodes()
    assert env.last_reached_nodes[0] == (3, 2)
    assert env.rejected_reached_node_updates == 1
