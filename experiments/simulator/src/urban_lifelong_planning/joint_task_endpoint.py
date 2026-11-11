"""Isolated, opt-in current-endpoint service-fact conditioning.

No experiment integration or historical-event replay. Exactness is relative to
selected service indicators and replay prior, NOT the complete public feed.
"""
from math import isfinite
from .component_exact_fast_reconstruction import FastExactComponentExecutionReconstructionEstimator
from .task_endpoint_conditioning import endpoint_constraints, filter_marginal_endpoint


class ZeroTaskLikelihood(ValueError):
    """No posterior exists on the available exact/empirical support."""


def condition_joint(states, ids, constraints):
    """Fresh joint dictionary plus retained mass; never invent support."""
    if any(not isfinite(w) or w < 0 for w in states.values()):
        raise ValueError('joint weights must be finite and nonnegative')
    kept = {s: w for s, w in states.items() if w > 0 and
            all(c.allows(s[k]) for k, i in enumerate(ids) for c in constraints[i])}
    mass = sum(kept.values())
    if mass <= 0:
        return {}, mass
    if not any(constraints[i] for i in ids):
        return dict(states), mass
    return {s: w / mass for s, w in kept.items()}, mass


class JointTaskEndpointEstimator(FastExactComponentExecutionReconstructionEstimator):
    method_name = 'RCSR-Component-Joint-Task-Endpoint-Prototype'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._task_constraints = None
        self.last_task_diagnostics = []

    def estimate(self, messages, paths_by_agent, actions_by_agent, current_step, *,
                 task_conditioning=False, active=(), completed=(), assignments=None,
                 service_radius=1.0, eligible=True):
        self.last_task_diagnostics = []
        self._task_constraints = None
        if task_conditioning:
            self.last_diagnostics = {}  # never expose a stale successful posterior on failure
            self._task_constraints = endpoint_constraints(tuple(active), tuple(completed),
                dict(assignments or {}), current_step, len(messages), service_radius, eligible=eligible)
        try:
            return super().estimate(messages, paths_by_agent, actions_by_agent, current_step)
        finally:
            # No posterior/history feedback to the next anchor replay.
            self._task_constraints = None

    def _condition(self, states, ids, mode, reason=None):
        if self._task_constraints is None:
            return states
        result, mass = condition_joint(states, ids, self._task_constraints)
        total = sum(states.values())
        # For equally weighted pre-filter particles and binary likelihoods,
        # importance-weight ESS equals the number of surviving particles, NOT
        # the reciprocal squared mass of the aggregated unique-state weights.
        ess = self.particle_count * mass / total if mode == 'particle' and total > 0 else None
        self.last_task_diagnostics.append(dict(agent_ids=tuple(ids), mode=mode,
            fallback_reason=reason, prior_support=len(states), posterior_support=len(result),
            constrained_agents=sum(bool(self._task_constraints[i]) for i in ids),
            retained_mass=mass, retained_fraction=mass/total if total else 0.,
            particle_ess=ess, particle_budget=self.particle_count if mode == 'particle' else None,
            zero_likelihood=not bool(result), posterior_valid=bool(result),
            temporal_scope='current_endpoint_only'))
        if not result:
            # The caller must count this failed reconstruction; it cannot be
            # labeled a posterior or silently replaced with incompatible states.
            raise ZeroTaskLikelihood(f'zero {mode} task likelihood for component {ids}')
        return result

    def _exact(self, ids, initial, commands):
        states = super()._exact(ids, initial, commands)
        return None if states is None else self._condition(states, ids, 'exact')

    def _sample(self, ids, initial, commands, anchor):
        states = super()._sample(ids, initial, commands, anchor)
        reason = 'agent_limit' if len(ids) > self.exact_agent_limit else 'state_limit'
        return self._condition(states, ids, 'particle', reason)


def matched_marginal_endpoint(beliefs, *, active=(), completed=(), assignments=None,
                              current_step, service_radius=1.0, eligible=True):
    """Same selected public evidence for original EMC and EMC-Medoid beliefs.

    Pure adapter: the original methods/decoders are unchanged. No environment
    accepted. Existing marginal fail-open behavior is explicitly flagged.
    """
    constraints = endpoint_constraints(tuple(active), tuple(completed), dict(assignments or {}),
        current_step, len(beliefs), service_radius, eligible=eligible)
    return tuple(filter_marginal_endpoint(b, cs) for b, cs in zip(beliefs, constraints))


def condition_for_control(estimator, unconditioned, messages, paths, actions, current_step, **public):
    """Opt-in runner boundary: whole-cycle control fallback, never a posterior.

    unconditioned must be the just-computed Component replay for this cycle.
    Catch only explicit zero-likelihood; programming/input errors propagate.
    No extra particles, truth injection or reuse of a previous cycle estimate.
    """
    try:
        if hasattr(estimator, 'condition_from_prior'):
            conditioned = estimator.condition_from_prior(unconditioned, current_step, **public)
        else:
            conditioned = estimator.estimate(messages, paths, actions, current_step,
                                             task_conditioning=True, **public)
    except ZeroTaskLikelihood:
        components = [dict(d) for d in estimator.last_task_diagnostics]
        return unconditioned, dict(conditional_posterior_valid=False,
            zero_evidence_count=sum(d['zero_likelihood'] for d in components),
            conditioning_fallback_count=1, joint_task_components=components,
            conditional_evidence_mass=None, conditional_true_state_probability=None,
            joint_task_control_source='unconditioned_component_whole_cycle',
            joint_task_temporal_scope='current_endpoint_only')
    components = [dict(d) for d in estimator.last_task_diagnostics]
    from math import prod
    return conditioned, dict(conditional_posterior_valid=True,
        zero_evidence_count=0, conditioning_fallback_count=0,
        joint_task_components=components,
        conditional_evidence_mass=prod(d['retained_fraction'] for d in components),
        # No truth enters inference; this diagnostic is intentionally unmeasured.
        conditional_true_state_probability=None,
        joint_task_control_source='conditioned_component',
        joint_task_temporal_scope='current_endpoint_only')
