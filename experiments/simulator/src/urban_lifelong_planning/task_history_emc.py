"""Opt-in independent EMC replay conditioned on timestamped service facts.

This is an independent per-agent comparator, not a joint collision posterior.
The caller derives the very same immutable facts as Component. No environment,
truth state, unissued actions, random tapes, or cached posterior is accepted.
"""
from math import prod

from .joint_task_endpoint import ZeroTaskLikelihood
from .rhcr_adapted import ExecutionModelConsistentEstimator, EMCStartEstimate


class TaskHistoryEMCEstimator(ExecutionModelConsistentEstimator):
    """Original EMC transitions and MAP tie rule with post-transition facts.

    Medoid decoding remains the caller's original supported medoid operation.
    Explicit off delegates exactly, including original missing-ledger behavior.
    Enabled history requires complete chronological issued commands because a
    skipped transition would apply service evidence to the wrong endpoint.
    """
    method_name = 'EMC-Task-History-Prototype'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_task_diagnostics = []

    def estimate(self, message, paths, actions, current_step, *,
                 history_conditioning=False, facts=()):
        self.last_task_diagnostics = []
        if not history_conditioning:
            return super().estimate(message, paths, actions, current_step)
        anchor = message.timestamp
        if anchor < 0 or anchor > current_step:
            raise ValueError('anchor must be a delivered historical state')
        relevant = tuple(a for a in actions if anchor < a.timestamp <= current_step)
        if [a.timestamp for a in relevant] != list(range(anchor + 1, current_step + 1)):
            raise ValueError('complete ordered post-anchor command history is required')
        for action in relevant:
            path = paths.get(action.path_id)
            if path is None or not 0 <= action.path_index < len(path):
                raise ValueError('issued command refers to an unavailable waypoint')
        by_step = {}
        for fact in tuple(facts):
            if fact.agent_id == message.drone_id and anchor < fact.timestamp <= current_step:
                by_step.setdefault(fact.timestamp, []).append(fact.constraint)

        belief = {message.position: 1.0}
        projection_mass = gap_mass = remaining_gap_mass = reference_mass = 0.0
        cumulative_evidence = 1.0
        for action in relevant:
            reference = paths[action.path_id][action.path_index]
            transition = self.transition(belief, reference, action.explicit_hover)
            belief = dict(transition.belief)
            # Original transition diagnostics are evaluated BEFORE this step's
            # observation. Later transitions naturally use conditioned belief.
            if not action.explicit_hover:
                action_mass = sum(belief.values())
                reference_mass += action_mass
                projection_mass += transition.reference_projection_rate * action_mass
                gap_mass += transition.mean_reference_gap * action_mass
                remaining_gap_mass += transition.mean_remaining_gap_after_projection * action_mass
            constraints = by_step.get(action.timestamp, ())
            total = sum(belief.values())
            kept = {point: mass for point, mass in belief.items()
                    if mass > 0 and all(c.allows(point) for c in constraints)}
            retained = sum(kept.values())
            valid = retained > 0
            fraction = retained / total if total else 0.0
            cumulative_evidence *= fraction
            self.last_task_diagnostics.append(dict(
                agent_ids=(message.drone_id,), timestamp=action.timestamp,
                mode='independent_exact', prior_support=sum(w > 0 for w in belief.values()),
                posterior_support=len(kept), constraint_count=len(constraints),
                fact_count=len(constraints), constrained_agents=int(bool(constraints)),
                retained_mass=retained, retained_fraction=fraction,
                conditional_evidence_mass=fraction, cumulative_evidence_mass=cumulative_evidence,
                particle_ess=None, zero_likelihood=not valid, posterior_valid=valid,
                temporal_scope='causal_window_history'))
            if not valid:
                raise ZeroTaskLikelihood(
                    f'zero independent EMC history likelihood for agent {message.drone_id} '
                    f'at timestamp {action.timestamp}')
            # No-constraint steps preserve original zero entries/FP behavior.
            if constraints:
                belief = {point: mass / retained for point, mass in kept.items()}
        position = max(belief, key=lambda p: (belief[p], -p[1], -p[0]))
        return EMCStartEstimate(position, tuple(sorted(belief.items())),
            projection_mass / max(reference_mass, 1e-12),
            gap_mass / max(reference_mass, 1e-12),
            remaining_gap_mass / max(reference_mass, 1e-12))


def condition_emc_history_for_control(estimator, unconditioned, messages,
                                      paths_by_agent, actions_by_agent, current_step,
                                      *, facts=()):
    """Whole-cycle invalid-posterior fallback to fresh original EMC values.

    Both MAP and medoid runner arms can consume these same estimates and use
    their original decoder. No partially conditioned fleet escapes on failure.
    Diagnostics cover attempted agents/steps only; unattempted ones are not
    implicitly counted as successful. Programming/input errors propagate.
    """
    n = len(messages)
    if not (n == len(unconditioned) == len(paths_by_agent) == len(actions_by_agent)):
        raise ValueError('matching fleet-length inputs required')
    if [m.drone_id for m in messages] != list(range(n)):
        raise ValueError('messages must be in consecutive agent-ID order')
    if len({m.timestamp for m in messages}) > 1:
        raise ValueError('mixed-timestamp asynchronous anchors are unsupported')
    facts = tuple(facts)
    results, diagnostics = [], []
    try:
        for i, message in enumerate(messages):
            try:
                results.append(estimator.estimate(message, paths_by_agent[i], actions_by_agent[i],
                    current_step, history_conditioning=True, facts=facts))
            finally:
                diagnostics.extend(dict(d) for d in estimator.last_task_diagnostics)
    except ZeroTaskLikelihood:
        return tuple(unconditioned), dict(
            conditional_posterior_valid=False, zero_evidence_count=1,
            conditioning_fallback_count=1, history_steps=diagnostics,
            conditional_evidence_mass=None,
            conditional_true_state_probability=None,
            history_control_source='unconditioned_emc_whole_cycle',
            history_temporal_scope='causal_window_history')
    return tuple(results), dict(
        conditional_posterior_valid=True, zero_evidence_count=0,
        conditioning_fallback_count=0, history_steps=diagnostics,
        conditional_evidence_mass=prod(d['retained_fraction'] for d in diagnostics),
        conditional_true_state_probability=None,
        history_control_source='conditioned_independent_emc',
        history_temporal_scope='causal_window_history')
