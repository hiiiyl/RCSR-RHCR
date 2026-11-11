"""Behavior-preserving exact-transition acceleration of component replay.

Legal projections/holds make edge revalidation redundant. This private kernel
must not be used for arbitrary proposals; the original scalar resolver remains
the reference for those. Probability iteration and arithmetic are unchanged.
"""
from collections import defaultdict
from itertools import product
from .component_fast_reconstruction import FastComponentExecutionReconstructionEstimator
from .rhcr_adapted import _diagonals_cross


def _resolve_legal_joint_moves(positions, proposals):
    """Ordered V1 safety filter with already legal proposals, including holds.

    Singletons have no safety interactions. For other sizes, keep crossing,
    start occupancy and accepted-end occupancy in the original order. Small
    tuple/list membership avoids constructing two sets per exact outcome.
    """
    if len(positions) == 1:
        return proposals
    accepted = []
    for i, proposal in enumerate(proposals):
        current = positions[i]
        if abs(proposal[0] - current[0]) == abs(proposal[1] - current[1]) == 1:
            for j, end in enumerate(accepted):
                if _diagonals_cross(current, proposal, positions[j], end):
                    proposal = current
                    break
        if proposal != current and proposal in positions:
            proposal = current
        if proposal in accepted:
            proposal = current
        accepted.append(proposal)
    return tuple(accepted)


class FastExactComponentExecutionReconstructionEstimator(FastComponentExecutionReconstructionEstimator):
    """Same exact posterior, overflow policy and sampled fallback as V2."""
    def _exact(self, ids, initial, commands):
        if len(ids) > self.exact_agent_limit:
            return None
        states = {tuple(initial[i] for i in ids): 1.}
        p = self.deviation_probability
        for frame in commands:
            updated = defaultdict(float)
            for state, mass in states.items():
                options = []
                for k, i in enumerate(ids):
                    reference, hover = frame[i]
                    end = state[k] if hover else self._project(state[k], reference)
                    if end == state[k] or p == 1.:
                        options.append(((state[k], 1.),))
                    elif p == 0.:
                        options.append(((end, 1.),))
                    else:
                        options.append(((state[k], p), (end, 1. - p)))
                for outcome in product(*options):
                    probability = mass
                    for _, weight in outcome:
                        probability *= weight
                    end = _resolve_legal_joint_moves(state, tuple(point for point, _ in outcome))
                    updated[end] += probability
                    if len(updated) > self.exact_state_limit:
                        return None
            states = updated
        return states
