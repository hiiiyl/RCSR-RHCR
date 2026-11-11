"""V2 causal component replay; exact bounded components, sampled remainder.

Components conservatively cover all positions reachable by legal tracker moves
or holds, including safety-induced holds. Shared cells or diagonal midpoints
merge components. Thus no safety interaction is possible across components.
No environment state, execution tape, or undelivered observation is accepted.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import product
import numpy as np

from .joint_reconstruction import (JointExecutionReconstructionEstimator,
    JointReconstructionEstimate, resolve_joint_moves, resolve_legal_joint_moves_batch)
from .motion_graph import legal_cell


class ComponentExecutionReconstructionEstimator(JointExecutionReconstructionEstimator):
    """Independent component joint MAP, with exact inference where affordable.

    Exactness is relative to the synchronized discrete tracker/hold model, not
    physical execution. ``joint_support_size`` counts the Cartesian product of
    component supports; ``particle_count`` retains the configured fallback
    sample budget (not an assertion that exact probabilities are empirical).
    ``last_diagnostics`` reports exact/sample components and support sizes.
    """
    method_name = "Component exact/particle execution replay V2"

    def __init__(self, *args, exact_agent_limit=3, exact_state_limit=512, **kwargs):
        super().__init__(*args, **kwargs)
        if exact_agent_limit < 0 or int(exact_agent_limit) != exact_agent_limit:
            raise ValueError("exact_agent_limit must be a nonnegative integer")
        if exact_state_limit < 1 or int(exact_state_limit) != exact_state_limit:
            raise ValueError("exact_state_limit must be a positive integer")
        self.exact_agent_limit = int(exact_agent_limit)
        self.exact_state_limit = int(exact_state_limit)
        self.last_diagnostics = {}
        self._hold_cache = {}

    def _holds(self, timestamp, agent):
        key = timestamp, agent
        if key not in self._hold_cache:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, timestamp, agent]))
            uniforms = (np.arange(self.particle_count) + rng.random(self.particle_count)) / self.particle_count
            rng.shuffle(uniforms)
            if len(self._hold_cache) >= 4096:
                self._hold_cache.clear()
            self._hold_cache[key] = uniforms < self.deviation_probability
        return self._hold_cache[key]

    def _components(self, initial, commands):
        parents = list(range(len(initial)))
        def root(i):
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i
        owners = {}
        for i, start in enumerate(initial):
            reachable = {start}
            cells = {start}
            midpoints = set()
            for frame in commands:
                reference, hover = frame[i]
                if hover:
                    continue
                next_cells = set(reachable)
                for position in reachable:
                    end = self._project(position, reference)
                    next_cells.add(end)
                    if abs(end[0] - position[0]) == 1 and abs(end[1] - position[1]) == 1:
                        midpoints.add((end[0] + position[0], end[1] + position[1]))
                reachable = next_cells
                cells.update(reachable)
            # Time-union footprints deliberately over-connect; never under-connect.
            for token in [('cell', p) for p in cells] + [('diagonal', p) for p in midpoints]:
                if token in owners:
                    parents[root(i)] = root(owners[token])
                else:
                    owners[token] = i
        groups = defaultdict(list)
        for i in range(len(initial)):
            groups[root(i)].append(i)
        return list(groups.values())

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
                    end = resolve_joint_moves(self.buildings, state, tuple(point for point, _ in outcome))
                    updated[end] += probability
                    if len(updated) > self.exact_state_limit:
                        # Restart from the public anchor with the same fallback RNG.
                        return None
            states = updated
        return states

    def _sample(self, ids, initial, commands, anchor):
        particles = np.tile(np.asarray([initial[i] for i in ids], dtype=np.int64), (self.particle_count, 1, 1))
        for timestamp, frame in enumerate(commands, anchor + 1):
            proposals = particles.copy()
            for k, i in enumerate(ids):
                reference, hover = frame[i]
                if hover:
                    continue
                moving = ~self._holds(timestamp, i)
                if not np.any(moving):
                    continue
                unique, inverse = np.unique(particles[moving, k, :], axis=0, return_inverse=True)
                projected = np.asarray([self._project(tuple(map(int, pos)), reference) for pos in unique])
                proposals[moving, k, :] = projected[inverse]
            particles = resolve_legal_joint_moves_batch(particles, proposals)
        states = defaultdict(float)
        for row in particles:
            states[tuple(tuple(map(int, pos)) for pos in row)] += 1. / self.particle_count
        return states

    def estimate(self, messages, paths_by_agent, actions_by_agent, current_step):
        n = len(messages)
        if n == 0 or len(paths_by_agent) != n or len(actions_by_agent) != n:
            raise ValueError("one nonempty telemetry/path/action collection is required per agent")
        if {m.drone_id for m in messages} != set(range(n)):
            raise ValueError("messages must contain each consecutive agent ID exactly once")
        messages = sorted(messages, key=lambda m: m.drone_id)
        if len({m.timestamp for m in messages}) != 1:
            raise ValueError("mixed-timestamp asynchronous anchors are unsupported")
        anchor = messages[0].timestamp
        if anchor < 0 or anchor > current_step:
            raise ValueError("anchor must be a delivered historical state")
        initial = tuple(m.position for m in messages)
        if len(set(initial)) != n or any(not legal_cell(self.buildings, p) for p in initial):
            raise ValueError("discrete joint anchor must contain distinct graph-valid cells")
        ledgers = []
        for actions in actions_by_agent:
            ledger = {}
            for action in actions:
                if anchor < action.timestamp <= current_step:
                    if action.timestamp in ledger:
                        raise ValueError("multiple commands for an agent at the same timestamp")
                    ledger[action.timestamp] = action
            ledgers.append(ledger)
        commands = []
        for timestamp in range(anchor + 1, current_step + 1):
            frame = []
            for i in range(n):
                action = ledgers[i].get(timestamp)
                if action is None:
                    raise ValueError("complete post-anchor command history is required for every agent")
                path = paths_by_agent[i].get(action.path_id)
                if path is None or not 0 <= action.path_index < len(path):
                    raise ValueError("issued command refers to an unavailable waypoint")
                frame.append((path[action.path_index], action.explicit_hover))
            commands.append(frame)
        positions, marginals, distributions = [None] * n, [None] * n, [None] * n
        support_size, exact_count = 1, 0
        components = self._components(initial, commands)
        supports = []
        for ids in components:
            states = self._exact(ids, initial, commands)
            if states is None:
                states = self._sample(ids, initial, commands, anchor)
            else:
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
        self.last_diagnostics = dict(component_sizes=[len(ids) for ids in components],
            exact_components=exact_count, sampled_components=len(components) - exact_count,
            component_support_sizes=supports)
        return JointReconstructionEstimate(tuple(positions), tuple(marginals), tuple(distributions), support_size, self.particle_count)
