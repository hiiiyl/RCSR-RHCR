"""Paired revision experiments with one common planning/execution loop.

Ledger timestamps denote the END of each issued action. State estimators see
delivered messages, the map, and issued commands; optional task endpoint
filtering additionally uses instantaneous public task notices (not delayed). True states are used
after planning for error measurement and inside the shared simulator only.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from math import hypot
from pathlib import Path
from time import perf_counter
import numpy as np

from .async_network_channel import AsyncNetworkChannel, generate_async_trace
from .delayed_state_channel import DelayedStateChannel
from .formal_tape_env import FormalTapeCityEnv
from .continuous_env import ContinuousLifelongCityEnv
from .continuous_dynamics import grid_to_continuous
from .lifelong_city_env import LifelongConfig
from .motion_graph import legal_transition
from .rhcr_adapted import (ExecutionProgressStartEstimator,
    ExecutionModelConsistentEstimator, IndexOnlyProgressStartEstimator,
    DeterministicReplayEstimator, IssuedPathAction, RHCRAdapted, RHCRConfig,
    PlanningFailure, select_joint_starts)


def supported_euclidean_medoid(belief):
    """Original Component decoder: risk, descending mass, y, then x ties."""
    points = np.asarray(list(belief))
    masses = np.asarray(list(belief.values()))
    risks = np.linalg.norm(points[:,None,:]-points[None,:,:], axis=2) @ masses
    idx = min(range(len(points)), key=lambda j: (risks[j], -masses[j], -points[j,1], -points[j,0]))
    return tuple(points[idx])


class TapeContinuousEnv(FormalTapeCityEnv, ContinuousLifelongCityEnv):
    def __init__(self, config: LifelongConfig, tape_path: Path):
        super().__init__(config, tape_path)
        self._init_continuous_state()


class SpatialReplayWithoutReachability:
    """Ablation retaining spatial aggregation, removing only the edge test."""
    physical_anchor_belief = True
    reference_tracker_model = "unconstrained_reference_jump"

    def estimate(self, message, paths, actions, current_step):
        belief = {message.position: 1.0}
        for action in actions:
            if not message.timestamp < action.timestamp <= current_step or action.explicit_hover:
                continue
            reference = paths[action.path_id][action.path_index]
            updated = {p: mass * .25 for p, mass in belief.items()}
            updated[reference] = updated.get(reference, 0.0) + .75 * sum(belief.values())
            belief = updated
        position = max(belief, key=lambda p: (belief[p], p[1], p[0]))
        return position, belief


@dataclass(frozen=True)
class EpisodeSpec:
    experiment: str
    method: str
    seed: int
    delay: int = 0
    condition: str = "fixed"
    fleet: int = 6
    commit_window: int = 2
    p_true: float = .25
    steps: int = 1000
    allocator_age_weight: float = 0.0
    allocator_use_estimates: bool = False
    goal_release: bool = False
    forbid_following: bool = False
    task_endpoint_conditioning: bool = False
    reference_policy: str = "nominal"
    observational_probability_trace: bool = False

    @property
    def key(self):
        base = f"{self.experiment}__{self.method}__s{self.seed}__d{self.delay}__{self.condition}__n{self.fleet}__k{self.commit_window}__p{self.p_true:g}__t{self.steps}"
        if self.allocator_age_weight or self.allocator_use_estimates:
            base += f"__aw{self.allocator_age_weight:g}__ae{int(self.allocator_use_estimates)}"
        if self.goal_release or self.forbid_following:
            base += f"__gr{int(self.goal_release)}__ff{int(self.forbid_following)}"
        if self.task_endpoint_conditioning:
            base += "__taskendpoint1"
        if self.reference_policy != "nominal":
            base += f"__ref{self.reference_policy}"
        return base


def run_episode(spec: EpisodeSpec, tape_path: Path):
    if spec.reference_policy not in ("nominal", "support_separated_forward_v1"):
        raise ValueError("Unknown reference controller")
    if spec.reference_policy != "nominal" and (spec.experiment == "continuous" or spec.condition != "fixed"):
        raise ValueError("Forward controller supports synchronized discrete telemetry only")
    history_methods = {
        "RCSR-Component-Task-History": "RCSR-Component-Bayes-V2",
        "EMC-Task-History": "EMC-RHCR",
        "EMC-Medoid-Task-History": "EMC-Bayes-Medoid",
    }
    history_enabled = spec.method in history_methods
    method = history_methods.get(spec.method, spec.method)
    if history_enabled and (spec.experiment == "continuous" or spec.condition != "fixed" or spec.task_endpoint_conditioning):
        raise ValueError("History conditioning requires synchronized discrete telemetry and cannot combine with endpoint filtering")

    cfg = LifelongConfig(seed=spec.seed, episode_steps=spec.steps,
        num_drones=spec.fleet, active_targets=2*spec.fleet, initial_energy=1500.)
    continuous = spec.experiment == "continuous"
    if spec.goal_release and (continuous or spec.condition != "fixed"):
        raise ValueError("Goal release currently supports synchronized discrete telemetry only")
    if spec.task_endpoint_conditioning and (continuous or spec.condition != "fixed" or method not in ("RCSR-Component-Bayes-V2", "EMC-Bayes-Medoid", "EMC-RHCR")):
        raise ValueError("Task endpoint filtering supports matched discrete fixed-telemetry Component/EMC methods only")
    if method == "RCSR-Component-Joint-Task-Endpoint" and (continuous or spec.condition != "fixed"):
        raise ValueError("Joint task endpoint conditioning requires synchronized discrete telemetry")
    env = (TapeContinuousEnv if continuous else FormalTapeCityEnv)(cfg, tape_path)
    from .fixed_allocator import FixedAllocator
    env.allocator = FixedAllocator(spec.allocator_age_weight)
    if spec.condition in ("moderate", "severe"):
        maximum, loss = (6, .1) if spec.condition == "moderate" else (10, .2)
        channel = AsyncNetworkChannel(spec.fleet, generate_async_trace(spec.seed, spec.fleet, spec.steps, maximum, loss, spec.condition))
    else:
        channel = DelayedStateChannel(spec.fleet, spec.delay)
    channel.bootstrap(env._state_messages(0, 0))
    horizon = spec.commit_window if method == "CC-RHCR" else 2
    planner = RHCRAdapted(RHCRConfig(planning_window=max(8, horizon), execution_horizon=horizon), forbid_following=spec.forbid_following)
    estimator = None
    joint_estimator = None
    if method in ("RCSR-Joint-Bayes", "RCSR-Component-Bayes-V2", "RCSR-Component-Joint-Task-Endpoint"):
        if continuous or spec.condition != "fixed":
            raise ValueError("Joint reconstruction currently supports synchronized discrete telemetry only")
        from .fast_joint_reconstruction import FastJointExecutionReconstructionEstimator as JointExecutionReconstructionEstimator
        if method == "RCSR-Component-Joint-Task-Endpoint":
            from .joint_task_endpoint_fast import SingleReplayJointTaskEndpointEstimator as JointTaskEndpointEstimator
            joint_estimator = JointTaskEndpointEstimator(env.building_mask, .25, 256)
        elif method == "RCSR-Component-Bayes-V2":
            from .component_exact_fast_reconstruction import FastExactComponentExecutionReconstructionEstimator as ComponentExecutionReconstructionEstimator
            joint_estimator = ComponentExecutionReconstructionEstimator(env.building_mask, .25, 256)
        else:
            joint_estimator = JointExecutionReconstructionEstimator(env.building_mask, .25, 256)
    if method == "RCSR-RHCR":
        estimator = ExecutionProgressStartEstimator(.25, buildings=env.building_mask)
    elif method in ("EMC-RHCR", "EMC-Bayes-Medoid"):
        estimator = ExecutionModelConsistentEstimator(env.building_mask, .25)
    elif method == "Index-only EP":
        estimator = IndexOnlyProgressStartEstimator(.25)
    elif method == "Deterministic replay":
        estimator = DeterministicReplayEstimator(env.building_mask)
    elif method == "Spatial replay no graph":
        estimator = SpatialReplayWithoutReachability()
    elif method not in ("RHCR-adapted", "CC-RHCR", "RCSR-Joint-Bayes", "RCSR-Component-Bayes-V2", "RCSR-Component-Joint-Task-Endpoint"):
        # Additional literature-derived backends are installed explicitly by
        # the runner; no generic method is silently substituted for a paper.
        from .literature_baselines import make_planner
        planner = make_planner(method, planner.config)
    if history_enabled:
        if method == "RCSR-Component-Bayes-V2":
            from .task_history import TaskHistoryComponentEstimator
            joint_estimator = TaskHistoryComponentEstimator(env.building_mask, .25, 256)
        else:
            from .task_history_emc import TaskHistoryEMCEstimator
            estimator = TaskHistoryEMCEstimator(env.building_mask, .25)
    # Eligibility concerns representation/physical reset, NOT support coverage.
    # Nominal plan-index ablations do not satisfy this contract.
    belief_provider = joint_estimator if joint_estimator is not None else estimator
    release_capable = belief_provider is None or getattr(belief_provider, "physical_anchor_belief", False)
    # Coordinate representation and tracker dynamics are separate contracts.
    # Every matched provider receives the identical controller; unmatched optional
    # requests stay nominal without changing the original estimator or goal release.
    from .belief_reference_control import SeparatedForwardReferenceControl
    from .reference_tracker_capability import reference_tracker_capability
    tracker_capability = reference_tracker_capability(belief_provider)
    reference_capable = release_capable and tracker_capability.compatible_forward
    reference_controller = (SeparatedForwardReferenceControl(env.building_mask)
        if spec.reference_policy != "nominal" and reference_capable else None)
    paths_by_id = [{} for _ in range(spec.fleet)]
    actions = [[] for _ in range(spec.fleet)]
    plan_id = -1
    cc_nominal = [m.position for m in channel.planner_states(0)]
    cc_last_timestamp = [-1] * spec.fleet
    start_run = perf_counter()
    while env.step_id < spec.steps:
        cycle_begin = perf_counter()
        channel.deliver(env.step_id)
        messages = channel.planner_states(env.step_id)
        delivered = [m.position for m in messages]
        reconstruct_begin = perf_counter()
        estimates, beliefs = [], []
        unconditioned_emc = []
        for i, message in enumerate(messages):
            if method == "CC-RHCR":
                if message.timestamp > cc_last_timestamp[i]:
                    cc_nominal[i] = message.position
                    cc_last_timestamp[i] = message.timestamp
                position, belief = cc_nominal[i], {cc_nominal[i]: 1.0}
            elif estimator is None:
                position, belief = message.position, {message.position: 1.0}
            else:
                value = estimator.estimate(message, paths_by_id[i], actions[i], env.step_id)
                if history_enabled:
                    unconditioned_emc.append(value)
                if isinstance(value, tuple):
                    position, belief = value
                else:
                    position = value.position
                    belief = {}
                    for point, mass in value.distribution:
                        belief[point] = belief.get(point, 0.0) + mass
            estimates.append(position)
            beliefs.append(belief)
        if joint_estimator is not None:
            joint = joint_estimator.estimate(messages, paths_by_id, actions, env.step_id)
            estimates = list(joint.marginal_positions)
            beliefs = [dict(b) for b in joint.distributions]
        task_constraints = [[] for _ in range(spec.fleet)]
        task_empty_support = 0
        task_retained_mass = [1.0] * spec.fleet
        pre_task_beliefs = [dict(b) for b in beliefs]
        pre_task_estimates = list(estimates)
        if spec.task_endpoint_conditioning and method in ("RCSR-Component-Bayes-V2", "EMC-Bayes-Medoid"):
            pre_task_estimates = [supported_euclidean_medoid(b) for b in beliefs]
        task_filter_begin = perf_counter()
        joint_task_audit = None
        history_audit = None
        if history_enabled:
            from .task_endpoint_conditioning import PublicTaskNotice
            from .task_history import history_for_control, window_history_facts
            def history_notice(t):
                return PublicTaskNotice(t.target_id, tuple(t.position), t.assigned_uav,
                                        t.assignment_step, t.completion_step)
            public = dict(active=tuple(history_notice(t) for t in env.target_pool.active.values()),
                completed=tuple(history_notice(t) for t in env.target_pool.completed),
                assignments=dict(env.assignments), service_radius=cfg.goal_radius)
            if method in ("RCSR-Component-Bayes-V2", "EMC-Bayes-Medoid"):
                pre_task_estimates = [supported_euclidean_medoid(b) for b in beliefs]
            if joint_estimator is not None:
                joint, history_audit = history_for_control(joint_estimator, joint,
                    messages, paths_by_id, actions, env.step_id, **public)
                beliefs = [dict(b) for b in joint.distributions]
            else:
                from .task_history_emc import condition_emc_history_for_control
                facts = window_history_facts(public['active'], public['completed'], public['assignments'],
                    messages[0].timestamp, env.step_id, spec.fleet, cfg.goal_radius)
                values, history_audit = condition_emc_history_for_control(estimator, unconditioned_emc,
                    messages, paths_by_id, actions, env.step_id, facts=facts)
                history_audit['history_fact_count'] = len(facts)
                beliefs = [dict(v.distribution) for v in values]
                estimates = [v.position for v in values]
        if method == "RCSR-Component-Joint-Task-Endpoint":
            from .joint_task_endpoint import condition_for_control
            from .task_endpoint_conditioning import PublicTaskNotice
            # Snapshot only the same public fields available to the marginal arms.
            def public_notice(t):
                return PublicTaskNotice(t.target_id, t.position, t.assigned_uav,
                                        t.assignment_step, t.completion_step)
            pre_task_estimates = [supported_euclidean_medoid(b) for b in beliefs]
            joint, joint_task_audit = condition_for_control(joint_estimator, joint,
                messages, paths_by_id, actions, env.step_id,
                active=tuple(public_notice(t) for t in env.target_pool.active.values()),
                completed=tuple(public_notice(t) for t in env.target_pool.completed),
                assignments=dict(env.assignments), service_radius=cfg.goal_radius)
            beliefs = [dict(b) for b in joint.distributions]
        if spec.task_endpoint_conditioning:
            from .task_endpoint_conditioning import PublicTaskNotice, endpoint_constraints, filter_marginal_endpoint
            # Explicit public-field boundary. No path_length, energy, positions,
            # execution records, or future tape data enter the helper.
            def notice(t):
                return PublicTaskNotice(t.target_id, t.position, t.assigned_uav,
                                        t.assignment_step, t.completion_step)
            task_constraints = endpoint_constraints(
                tuple(notice(t) for t in env.target_pool.active.values()),
                tuple(notice(t) for t in env.target_pool.completed),
                dict(env.assignments), env.step_id, spec.fleet, cfg.goal_radius)
            for i in range(spec.fleet):
                beliefs[i], task_retained_mass[i], empty = filter_marginal_endpoint(beliefs[i], task_constraints[i])
                task_empty_support += int(empty)
            if method == "EMC-RHCR":
                estimates = [max(b, key=lambda p: (b[p], -p[1], -p[0])) for b in beliefs]
        if method in ("RCSR-Joint-Bayes", "RCSR-Component-Bayes-V2", "EMC-Bayes-Medoid", "RCSR-Component-Joint-Task-Endpoint"):
            estimates = [supported_euclidean_medoid(belief) for belief in beliefs]
        task_filter_ms = 1000 * (perf_counter() - task_filter_begin) if spec.task_endpoint_conditioning or joint_task_audit is not None or history_audit is not None else 0.0
        raw_estimates = list(estimates)
        joint_repair, failed = False, False
        try:
            if len(set(estimates)) != spec.fleet:
                estimates = select_joint_starts(beliefs, env.building_mask)
                joint_repair = True
        except PlanningFailure:
            failed = True
        reconstruction_ms = 1000*(perf_counter()-reconstruct_begin)
        allocation_begin = perf_counter()
        # Unified optional interface: each method uses its own causally valid
        # reconstructed starts; the delivered-state ablation remains default.
        env._allocate_idle(estimates if spec.allocator_use_estimates else delivered)
        goals = [env.target_pool.active[env.assignments[i]].position if i in env.assignments else estimates[i] for i in range(spec.fleet)]
        allocation_ms = 1000*(perf_counter()-allocation_begin)
        search_begin = perf_counter()
        candidate_paths = None
        if not failed:
            try:
                candidate_paths = planner.plan(env.building_mask, estimates, goals)
            except PlanningFailure:
                failed = True
        release = [False] * spec.fleet
        if spec.goal_release and release_capable and not failed and not planner.last_plan_fallback:
            from .belief_goal_release import goal_release_mask
            release = goal_release_mask(beliefs, goals, candidate_paths, set(env.assignments), horizon)
        search_ms = 1000*(perf_counter()-search_begin)
        reference_begin = perf_counter()
        selected_indices = None
        forward_diagnostics = {}
        if reference_controller is not None and not failed and not planner.last_plan_fallback:
            selected_indices, selected_holds, forward_diagnostics = reference_controller.choose(
                beliefs, goals, candidate_paths, set(env.assignments), release, horizon)
        reference_control_ms = 1000*(perf_counter()-reference_begin)
        planning_total_ms = 1000*(perf_counter()-cycle_begin)
        record = {"step":env.step_id, "planning_time_ms":search_ms,
            "reconstruction_time_ms":reconstruction_ms, "allocation_time_ms":allocation_ms,
            "total_cycle_time_ms":planning_total_ms, "joint_repair":int(joint_repair),
            "duplicate_raw_starts":spec.fleet-len(set(raw_estimates)),
            "goal_release_agent_count":sum(release),
            "goal_release_capable":int(release_capable),
            "goal_release_incompatible_request":int(spec.goal_release and not release_capable),
            "planning_failure":int(failed),
            "planner_wait_fallback":int(not failed and getattr(planner,"last_plan_fallback",False)),
            "mean_planning_start_error":float(np.mean([hypot(p[0]-q[0],p[1]-q[1]) for p,q in zip(estimates,env.positions)])),
            "mean_raw_start_error":float(np.mean([hypot(p[0]-q[0],p[1]-q[1]) for p,q in zip(raw_estimates,env.positions)])),
            "true_support_coverage":float(np.mean([b.get(tuple(q), 0.) > 0. for b,q in zip(beliefs,env.positions)])),
            "mean_support_size":float(np.mean([len(b) for b in beliefs])),
            "mean_aoi":float(np.mean([env.step_id-m.timestamp for m in messages])),
            "planner_state_max_timestamp":max(m.timestamp for m in messages)}
        record.update(forward_diagnostics)
        record.update({
            "reference_control_time_ms": reference_control_ms,
            "reference_control_capable": int(reference_capable),
            "reference_physical_coordinates": int(tracker_capability.physical_coordinates),
            "reference_tracker_compatible": int(tracker_capability.compatible_forward),
            "reference_tracker_model": tracker_capability.model,
            "reference_tracker_incompatibility_reason": tracker_capability.reason,
            "reference_control_incompatible_request": int(spec.reference_policy != "nominal" and not reference_capable),
            "effective_reference_controller": spec.reference_policy if reference_controller is not None else "nominal",
            "runtime_phase": "planning_control_before_execution",
            "issued_path_indices": [], "issued_holds": [], "issued_references": [],
            "issued_actions": [], "attained_state_messages": [],
        })
        # Observational snapshots are downstream of the complete decision. They
        # are never supplied to a controller, estimator, planner, or allocator.
        record["delivered_state_messages"] = [asdict(m) for m in messages]
        if spec.observational_probability_trace:
            record["prior_distributions"] = [list(b.items()) for b in pre_task_beliefs]
            record["control_distributions"] = [list(b.items()) for b in beliefs]
        record["control_starts"] = list(estimates)
        record["control_goals"] = list(goals)
        if spec.task_endpoint_conditioning:
            # Post-decision truth diagnostics are never fed back to filtering.
            record.update({
                "task_endpoint_filter_and_decode_time_ms": task_filter_ms,
                "task_endpoint_before_filter_raw_error": float(np.mean([hypot(p[0]-q[0], p[1]-q[1]) for p,q in zip(pre_task_estimates,env.positions)])),
                "task_endpoint_after_filter_raw_error": float(np.mean([hypot(p[0]-q[0], p[1]-q[1]) for p,q in zip(raw_estimates,env.positions)])),
                "task_endpoint_final_control_start_error": record["mean_planning_start_error"],
                "task_endpoint_constrained_agent_count": sum(bool(cs) for cs in task_constraints),
                "task_endpoint_constraint_count": sum(map(len, task_constraints)),
                "task_endpoint_empty_support_count": task_empty_support,
                "task_endpoint_mean_retained_mass": float(np.mean(task_retained_mass)),
                "task_endpoint_true_constraint_exclusions": sum(not all(c.allows(q) for c in cs) for q, cs in zip(env.positions, task_constraints)),
                "task_endpoint_true_support_removed": sum(b.get(q, 0.) > 0 and a.get(q, 0.) <= 0 for q, b, a in zip(env.positions, pre_task_beliefs, beliefs)),
                "task_endpoint_pre_true_support_coverage": float(np.mean([b.get(q, 0.) > 0 for q, b in zip(env.positions, pre_task_beliefs)])),
            })
        if joint_task_audit is not None:
            record.update(joint_task_audit)
            record.update({
                "joint_task_filter_and_decode_time_ms": task_filter_ms,
                "joint_task_before_filter_raw_error": float(np.mean([hypot(p[0]-q[0],p[1]-q[1]) for p,q in zip(pre_task_estimates,env.positions)])),
                # This is a CONTROL distribution/error, not a conditional posterior if invalid.
                "joint_task_control_raw_error": record["mean_raw_start_error"],
                "joint_task_final_control_start_error": record["mean_planning_start_error"],
            })
        if history_audit is not None:
            record.update(history_audit)
            record.update({
                "history_filter_and_decode_time_ms": task_filter_ms,
                "history_before_filter_raw_error": float(np.mean([hypot(p[0]-q[0],p[1]-q[1]) for p,q in zip(pre_task_estimates,env.positions)])),
                "history_after_filter_raw_error": record["mean_raw_start_error"] if history_audit['conditional_posterior_valid'] else None,
                "history_control_raw_error": record["mean_raw_start_error"],
                "history_final_control_start_error": record["mean_planning_start_error"],
                "history_pre_true_support_coverage": float(np.mean([b.get(q,0.) > 0 for b,q in zip(pre_task_beliefs,env.positions)])),
                "history_post_true_support_coverage": record['true_support_coverage'] if history_audit['conditional_posterior_valid'] else None,
            })
        if hasattr(planner, 'last_route_generated_count'):
            record['route_generated_count'] = 0 if failed else planner.last_route_generated_count
            record['route_fallback_count'] = 0 if failed else planner.last_route_fallback_count
        if continuous:
            record['mean_continuous_start_error_m']=float(np.mean([hypot(grid_to_continuous(*p,env.dyn_config.cell_size)[0]-q[0],grid_to_continuous(*p,env.dyn_config.cell_size)[1]-q[1]) for p,q in zip(estimates,env._continuous_positions)]))
        env.planning_records.append(record)
        plan_id += 1
        execute_count = min(1 if failed else horizon, spec.steps-env.step_id)
        if failed:
            # These paths are command-ledger placeholders only, never returned
            # or described as feasible plans. The command is an explicit hold.
            candidate_paths = [[p,p] for p in estimates]
        record["issued_plan_id"] = plan_id
        record["nominal_paths"] = candidate_paths
        for i,path in enumerate(candidate_paths):
            paths_by_id[i][plan_id] = path
        if method == "CC-RHCR" and not failed:
            cc_nominal = [path[horizon] for path in candidate_paths]
        reached_indices = [0]*spec.fleet
        for j in range(1,execute_count+1):
            indices = selected_indices[j-1] if selected_indices is not None else [j]*spec.fleet
            references = [path[indices[i]] for i,path in enumerate(candidate_paths)]
            holds = [failed or (path[j] == path[j-1] and not release[i]) for i,path in enumerate(candidate_paths)]
            record["issued_path_indices"].append(list(indices))
            record["issued_holds"].append(list(holds))
            record["issued_references"].append(list(references))
            issued = [IssuedPathAction(env.step_id+1,plan_id,indices[i],holds[i]) for i in range(spec.fleet)]
            record["issued_actions"].append([asdict(a) for a in issued])
            for i in range(spec.fleet):
                actions[i].append(issued[i])
            env._execute_controlled_step(references,spec.p_true,hold_mask=holds)
            for i in range(spec.fleet):
                if not holds[i] and env.positions[i] == references[i]:
                    reached_indices[i] = indices[i]
            attained = env._state_messages(env.step_id,j,plan_id,reached_indices)
            record["attained_state_messages"].append([asdict(m) for m in attained])
            channel.send(attained)
            channel.deliver(env.step_id)
        # Delivered anchors make older commands irrelevant. Retain only the
        # history still needed by that UAV; no execution outcomes enter replay.
        for i,message in enumerate(messages):
            actions[i] = [a for a in actions[i] if a.timestamp > message.timestamp]
            needed = {a.path_id for a in actions[i]} | {message.path_id, plan_id}
            paths_by_id[i] = {k:p for k,p in paths_by_id[i].items() if k in needed}
    summary = env.summary()
    if continuous:
        summary.update(env.continuous_summary())
    warmup = min(100, spec.steps//10)
    summary.update(vars(spec))
    if spec.task_endpoint_conditioning:
        for name in ("task_endpoint_constrained_agent_count", "task_endpoint_constraint_count", "task_endpoint_empty_support_count", "task_endpoint_true_constraint_exclusions", "task_endpoint_true_support_removed"):
            summary[name] = sum(r[name] for r in env.planning_records)
        for name in ("task_endpoint_filter_and_decode_time_ms", "task_endpoint_before_filter_raw_error", "task_endpoint_after_filter_raw_error", "task_endpoint_final_control_start_error"):
            summary["mean_" + name] = float(np.mean([r[name] for r in env.planning_records]))
        denominator = summary["task_endpoint_constrained_agent_count"]
        summary["task_endpoint_empty_support_rate"] = summary["task_endpoint_empty_support_count"] / denominator if denominator else None
        summary["task_endpoint_inference"] = "independent marginal endpoint filtering; not joint posterior"
        summary["task_endpoint_notice_assumption"] = "instantaneous public tasks; delayed position telemetry"
    if method == "RCSR-Component-Joint-Task-Endpoint":
        for name in ("zero_evidence_count", "conditioning_fallback_count"):
            summary[name] = sum(r[name] for r in env.planning_records)
        summary["conditional_posterior_valid"] = all(r["conditional_posterior_valid"] for r in env.planning_records)
        summary["joint_task_inference"] = "component joint current-endpoint selected service facts; exact/particle mixed; no history"
        summary["joint_task_fallback"] = "whole-cycle unconditioned Component control; not conditional posterior"
        summary["joint_task_notice_assumption"] = "instantaneous public tasks; delayed position telemetry"
        summary["joint_task_component_records"] = sum(len(r["joint_task_components"]) for r in env.planning_records)
        summary["joint_task_sampled_component_records"] = sum(d["mode"] == "particle" for r in env.planning_records for d in r["joint_task_components"])
        for name in ("joint_task_filter_and_decode_time_ms", "joint_task_before_filter_raw_error", "joint_task_control_raw_error", "joint_task_final_control_start_error"):
            summary["mean_" + name] = float(np.mean([r[name] for r in env.planning_records]))
    if history_enabled:
        for name in ("zero_evidence_count", "conditioning_fallback_count", "history_fact_count"):
            summary[name] = sum(r[name] for r in env.planning_records)
        summary['conditional_posterior_valid'] = all(r['conditional_posterior_valid'] for r in env.planning_records)
        summary['history_inference'] = ('component exact/particle' if joint_estimator is not None else 'independent exact EMC') + '; selected causal window service facts'
        summary['history_notice_assumption'] = 'instantaneous public task notices; delayed synchronized state telemetry'
        summary['history_cost_disclosure'] = 'two replay passes including prior control fallback and before-filter decode'
        for name in ('history_filter_and_decode_time_ms', 'history_before_filter_raw_error', 'history_after_filter_raw_error', 'history_control_raw_error', 'history_final_control_start_error'):
            values = [r[name] for r in env.planning_records if r[name] is not None]
            summary['mean_' + name] = float(np.mean(values)) if values else None
    summary["effective_reference_controller"] = spec.reference_policy if reference_controller is not None else "nominal"
    summary["reference_control_capable"] = bool(reference_capable)
    summary["reference_physical_coordinates"] = tracker_capability.physical_coordinates
    summary["reference_tracker_compatible"] = tracker_capability.compatible_forward
    summary["reference_tracker_model"] = tracker_capability.model
    summary["reference_tracker_incompatibility_reason"] = tracker_capability.reason
    summary["reference_control_incompatible_request"] = bool(spec.reference_policy != "nominal" and not reference_capable)
    summary["runtime_phase"] = "planning_control_before_execution; episode runtime includes execution and observation"
    for name in ("forward_eligible_agents", "forward_changed_references", "forward_expected_distance_gain"):
        summary[name] = sum(r.get(name, 0) for r in env.planning_records)
    summary["mean_reference_control_time_ms"] = float(np.mean([r["reference_control_time_ms"] for r in env.planning_records]))
    summary["goal_release_capable"] = bool(release_capable)
    summary["estimator_implementation"] = type(belief_provider).__name__ if belief_provider is not None else "point_estimate_adapter"
    summary['episode_key']=spec.key
    summary['runtime_s']=perf_counter()-start_run
    summary['steady_state_throughput']=sum(t.completion_step>warmup for t in env.target_pool.completed)/(spec.steps-warmup)
    summary['safety_interventions_per_step']=summary['safety_intervention_count']/spec.steps
    summary['safety_interventions_per_uav_step']=summary['safety_intervention_count']/(spec.steps*spec.fleet)
    if not env.target_pool.completed:
        summary['mean_response_time']=None
        summary['p90_response_time']=None
    for name in ('reconstruction_time_ms','allocation_time_ms','total_cycle_time_ms','mean_planning_start_error','mean_raw_start_error','mean_support_size','true_support_coverage','mean_aoi','mean_continuous_start_error_m'):
        values=[r[name] for r in env.planning_records if name in r]
        if values:
            summary['mean_'+name if not name.startswith('mean_') else name]=float(np.mean(values))
            summary['p95_'+name]=float(np.percentile(values,95))
    for name in ('joint_repair','duplicate_raw_starts','planning_failure','planner_wait_fallback','goal_release_agent_count','goal_release_incompatible_request'):
        summary[name+'_count']=sum(r[name] for r in env.planning_records)
    if hasattr(planner, 'last_route_generated_count'):
        for name in ('route_generated_count', 'route_fallback_count'):
            summary[name] = sum(r.get(name, 0) for r in env.planning_records)
        route_total = summary['route_generated_count'] + summary['route_fallback_count']
        summary['route_fallback_fraction'] = summary['route_fallback_count'] / route_total if route_total else None
    summary['invalid_graph_transition_count']=sum(not legal_transition(env.building_mask,a,b) for path in env.trajectories for a,b in zip(path,path[1:]))
    tasks=[]
    for task in list(env.target_pool.completed)+list(env.target_pool.active.values()):
        row=task.as_dict()
        row.update({'event_observed':int(task.completion_step is not None),
                    'observed_duration':(task.completion_step if task.completion_step is not None else spec.steps)-task.release_step})
        tasks.append(row)
    # A common release cohort (the initial 2N tasks) permits a fair comparison
    # with complete fixed-horizon censoring, unlike completed-only P90.
    initial=[r for r in tasks if r['target_id']<2*spec.fleet]
    summary['initial_cohort_completion_fraction']=float(np.mean([r['event_observed'] for r in initial]))
    summary['initial_cohort_restricted_wait_steps']=float(np.mean([r['observed_duration'] for r in initial]))
    summary['total_observed_waiting_steps']=sum(r['observed_duration'] for r in tasks)
    return summary,tasks,env.planning_records
