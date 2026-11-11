"""Optional causal service-history filtering of the original component replay.

Exactness is relative to the tracker/hold prior and selected public service
facts. No environment, execution outcomes, future packets or random tape enter.
"""
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from math import isfinite, prod
from numbers import Real
import numpy as np

from .component_exact_fast_reconstruction import (
    FastExactComponentExecutionReconstructionEstimator, _resolve_legal_joint_moves)
from .fast_joint_reconstruction import resolve_legal_joint_moves_batch_fast
from .joint_task_endpoint import ZeroTaskLikelihood, condition_joint
from .task_endpoint_conditioning import EndpointConstraint


@dataclass(frozen=True)
class HistoryFact:
    timestamp: int
    agent_id: int
    target_id: int
    constraint: EndpointConstraint


def window_history_facts(active, completed, assignments, anchor, current_step,
                         fleet, radius=1., *, eligible=True):
    """Immutable service facts on (delivered anchor, current], before allocation.

    Unknown assignee/timestamps and future records supply no facts. Known
    inconsistent identities or overlapping service schedules are errors, not
    permission to drop contradictory evidence. Same-coordinate distinct IDs
    are valid. Exact duplicate records are deduplicated by identity.
    """
    if not 0 <= anchor <= current_step or fleet < 1:
        raise ValueError('invalid history window/fleet')
    if not isfinite(radius) or radius < 0:
        raise ValueError('invalid service radius')
    if not eligible:
        return ()
    records = {}; intervals = defaultdict(list); result = []
    for kind, notices in (('active', tuple(active)), ('completed', tuple(completed))):
        for task in notices:
            key = task.target_id
            item = (kind, task)
            if key in records:
                if records[key] != item:
                    raise ValueError('conflicting public records for one target identity')
                continue
            records[key] = item
            i, a, c = task.assigned_uav, task.assignment_step, task.completion_step
            if i is None or a is None or (kind == 'completed' and c is None):
                continue
            if not isinstance(i, int) or not 0 <= i < fleet:
                raise ValueError('invalid public assignee')
            if not isinstance(a, int) or a < 0 or (c is not None and (not isinstance(c, int) or c <= a)):
                raise ValueError('invalid known service chronology')
            if a > current_step or (c is not None and c > current_step):
                continue
            if (not isinstance(task.position, (tuple, list)) or len(task.position) != 2
                    or any(not isinstance(v, Real) or not isfinite(v) for v in task.position)):
                raise ValueError('public task position must contain two finite real coordinates')
            if kind == 'active':
                if c is not None:
                    raise ValueError('completed timestamp on active record')
                if assignments.get(i) != key:
                    raise ValueError('active record and public assignment disagree')
                end = current_step
            else:
                end = c
                if assignments.get(i) == key:
                    raise ValueError('completed task remains currently assigned')
            intervals[i].append((a, end, key))
            for timestamp in range(max(anchor, a) + 1, end + 1):
                result.append(HistoryFact(timestamp, i, key,
                    EndpointConstraint(tuple(task.position), radius, kind == 'completed' and timestamp == c)))
    # Assignment at the previous completion endpoint is legal, but an active
    # task cannot have been replaced before completing in this audited model.
    for spans in intervals.values():
        spans.sort()
        for left, right in zip(spans, spans[1:]):
            if right[0] < left[1]:
                raise ValueError('overlapping public assignment intervals')
    return tuple(sorted(result, key=lambda f: (f.timestamp, f.agent_id, f.target_id)))


class TaskHistoryComponentEstimator(FastExactComponentExecutionReconstructionEstimator):
    method_name = 'RCSR-Component-Task-History-Prototype'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._history = None
        self._anchor = None
        self.last_task_diagnostics = []
        self.last_history_facts = ()

    def estimate(self, messages, paths_by_agent, actions_by_agent, current_step, *,
                 history_conditioning=False, active=(), completed=(), assignments=None,
                 service_radius=1., eligible=True):
        self.last_task_diagnostics = []
        self.last_history_facts = ()
        self._history = None
        if history_conditioning:
            if not messages or len({m.timestamp for m in messages}) != 1:
                raise ValueError('synchronized delivered anchors required')
            self.last_diagnostics = {}
            self._anchor = messages[0].timestamp
            self.last_history_facts = window_history_facts(tuple(active), tuple(completed),
                dict(assignments or {}), self._anchor, current_step, len(messages),
                service_radius, eligible=eligible)
            self._history = defaultdict(list)
            for fact in self.last_history_facts:
                self._history[fact.timestamp].append(fact)
        try:
            return super().estimate(messages, paths_by_agent, actions_by_agent, current_step)
        finally:
            self._history = None
            self._anchor = None

    def _constraints(self, timestamp, fleet):
        constraints = [[] for _ in range(fleet)]
        for fact in self._history.get(timestamp, ()):
            constraints[fact.agent_id].append(fact.constraint)
        return constraints

    def _record(self, ids, timestamp, mode, prior, posterior, fraction,
                cumulative, constraints, *, reason=None, survivors=None):
        record = dict(agent_ids=tuple(ids), timestamp=timestamp, mode=mode,
            fallback_reason=reason, prior_support=prior, posterior_support=posterior,
            constrained_agents=sum(bool(constraints[i]) for i in ids),
            fact_count=sum(len(constraints[i]) for i in ids),
            conditional_evidence_mass=fraction, retained_fraction=fraction,
            cumulative_evidence_mass=cumulative,
            particle_ess=survivors, particle_budget=self.particle_count if mode == 'particle' else None,
            posterior_valid=posterior > 0, zero_likelihood=posterior == 0,
            temporal_scope='causal_window_history')
        self.last_task_diagnostics.append(record)
        if posterior == 0:
            raise ZeroTaskLikelihood(f'zero {mode} history likelihood at {timestamp} for {ids}')

    def _exact(self, ids, initial, commands):
        if self._history is None:
            return super()._exact(ids, initial, commands)
        if len(ids) > self.exact_agent_limit:
            return None
        diagnostic_start = len(self.last_task_diagnostics)
        states = {tuple(initial[i] for i in ids): 1.}
        p = self.deviation_probability
        cumulative = 1.
        for timestamp, frame in enumerate(commands, self._anchor + 1):
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
                    # Deliberately BEFORE this endpoint's evidence. Earlier
                    # history filtering can still change the later overflow.
                    if len(updated) > self.exact_state_limit:
                        del self.last_task_diagnostics[diagnostic_start:]
                        return None
            cs = self._constraints(timestamp, len(initial))
            states, fraction = condition_joint(updated, ids, cs)
            cumulative *= fraction
            self._record(ids, timestamp, 'exact', len(updated), len(states), fraction, cumulative, cs)
        return states

    def _sample(self, ids, initial, commands, anchor):
        if self._history is None:
            return super()._sample(ids, initial, commands, anchor)
        reason = 'agent_limit' if len(ids) > self.exact_agent_limit else 'state_limit'
        particles = np.tile(np.asarray([initial[i] for i in ids], dtype=np.int64), (self.particle_count, 1, 1))
        alive = np.ones(self.particle_count, dtype=bool)
        height = self.buildings.shape[0]
        for timestamp, frame in enumerate(commands, anchor + 1):
            proposals = particles.copy()
            for k, i in enumerate(ids):
                reference, hover = frame[i]
                if hover:
                    continue
                moving = ~self._holds(timestamp, i)
                if not np.any(moving):
                    continue
                cells = particles[moving, k, :]
                unique, inverse = np.unique(cells[:, 0] * height + cells[:, 1], return_inverse=True)
                projected = np.asarray([self._project((int(cell // height), int(cell % height)), reference) for cell in unique])
                proposals[moving, k, :] = projected[inverse]
            particles = resolve_legal_joint_moves_batch_fast(particles, proposals)
            before = int(alive.sum())
            prior_support = len({tuple(map(tuple, row)) for row in particles[alive]})
            cs = self._constraints(timestamp, len(initial))
            for j in np.flatnonzero(alive):
                alive[j] = all(c.allows(particles[j, k]) for k, i in enumerate(ids) for c in cs[i])
            survivors = int(alive.sum())
            posterior_support = len({tuple(map(tuple, row)) for row in particles[alive]})
            self._record(ids, timestamp, 'particle', prior_support, posterior_support,
                survivors / before, survivors / self.particle_count, cs,
                reason=reason, survivors=survivors)
        states = defaultdict(float)
        for row in particles[alive]:
            states[tuple(tuple(map(int, pos)) for pos in row)] += 1. / int(alive.sum())
        return states


def history_for_control(estimator, unconditioned, messages, paths, actions, current_step, **public):
    """Fresh same-cycle prior is a control fallback, never an invalid posterior."""
    valid = True
    try:
        result = estimator.estimate(messages, paths, actions, current_step,
                                    history_conditioning=True, **public)
    except ZeroTaskLikelihood:
        result, valid = unconditioned, False
    records = [dict(d) for d in estimator.last_task_diagnostics]
    return result, dict(conditional_posterior_valid=valid,
        zero_evidence_count=sum(d['zero_likelihood'] for d in records),
        conditioning_fallback_count=int(not valid), history_steps=records,
        conditional_evidence_mass=prod(d['retained_fraction'] for d in records) if valid else None,
        conditional_true_state_probability=None,
        history_fact_count=len(estimator.last_history_facts),
        history_control_source='conditioned_component' if valid else 'unconditioned_component_whole_cycle',
        history_temporal_scope='causal_window_history')
