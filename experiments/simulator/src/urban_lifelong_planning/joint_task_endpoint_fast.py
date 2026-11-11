"""Single-replay adapter for current-endpoint conditioning, not a new model.

The prior pass captures the same ordered prefilter dictionaries used for its
marginals. Conditioning consumes that prior by identity, without replaying any
transitions or drawing particles. The reference estimate(task_conditioning=True)
path remains inherited and available for differential checks.
"""
from collections import defaultdict
from .joint_task_endpoint import JointTaskEndpointEstimator
from .joint_reconstruction import JointReconstructionEstimate
from .task_endpoint_conditioning import endpoint_constraints


class SingleReplayJointTaskEndpointEstimator(JointTaskEndpointEstimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._capturing = False
        self._prior = None
        self._prior_step = None
        self._prefilter = []

    def estimate(self, messages, paths_by_agent, actions_by_agent, current_step, **kwargs):
        self._prior = None
        self._prefilter = []
        self._capturing = not kwargs.get('task_conditioning', False)
        try:
            result = super().estimate(messages, paths_by_agent, actions_by_agent,
                                      current_step, **kwargs)
            if self._capturing:
                self._prior, self._prior_step = result, current_step
            return result
        finally:
            self._capturing = False

    def _exact(self, ids, initial, commands):
        states = super()._exact(ids, initial, commands)
        if self._capturing and states is not None:
            self._prefilter.append((tuple(ids), states, 'exact', None))
        return states

    def _sample(self, ids, initial, commands, anchor):
        states = super()._sample(ids, initial, commands, anchor)
        if self._capturing:
            reason = 'agent_limit' if len(ids) > self.exact_agent_limit else 'state_limit'
            self._prefilter.append((tuple(ids), states, 'particle', reason))
        return states

    def condition_from_prior(self, prior, current_step, *, active=(), completed=(),
                             assignments=None, service_radius=1.0, eligible=True):
        """Condition exactly this estimator's latest successful prior.

        All components were computed before conditioning. On failure only the
        attempted prefix is assessed, exactly as in the two-replay reference.
        Repeated conditioning of this immutable prior is allowed; no posterior
        is cached and no historical evidence is fed back into later estimates.
        """
        if prior is not self._prior or current_step != self._prior_step:
            raise ValueError('conditioning requires the latest same-cycle prior')
        self.last_task_diagnostics = []
        self.last_diagnostics = {}
        n = len(prior.distributions)
        self._task_constraints = endpoint_constraints(tuple(active), tuple(completed),
            dict(assignments or {}), current_step, n, service_radius, eligible=eligible)
        try:
            positions, marginals, distributions = [None] * n, [None] * n, [None] * n
            support_size, exact_count = 1, 0
            supports = []
            for ids, raw, mode, reason in self._prefilter:
                states = self._condition(raw, ids, mode, reason)
                if mode == 'exact':
                    exact_count += 1
                support_size *= len(states)
                supports.append(len(states))
                joint_map = max(states, key=lambda state: (states[state], state))
                for k, i in enumerate(ids):
                    counts = defaultdict(float)
                    for state, mass in states.items():
                        counts[state[k]] += mass
                    positions[i] = joint_map[k]
                    marginals[i] = max(counts, key=lambda pos: (counts[pos], pos[1], pos[0]))
                    distributions[i] = tuple(sorted(counts.items()))
            self.last_diagnostics = dict(component_sizes=[len(ids) for ids, *_ in self._prefilter],
                exact_components=exact_count, sampled_components=len(self._prefilter) - exact_count,
                component_support_sizes=supports)
            return JointReconstructionEstimate(tuple(positions), tuple(marginals),
                tuple(distributions), support_size, self.particle_count)
        finally:
            self._task_constraints = None
