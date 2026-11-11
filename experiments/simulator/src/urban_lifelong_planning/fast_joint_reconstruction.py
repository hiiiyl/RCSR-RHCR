"""Behavior-preserving acceleration of V1 stratified joint particle replay.

The original implementation remains an independent reference oracle. Random
holds use the original RNG, generation and shuffle order. Cache keys include
all public random parameters; bounded eviction affects performance only.
"""
from __future__ import annotations
from collections import Counter, OrderedDict
import numpy as np
from .delayed_state_channel import StateMessage
from .motion_graph import legal_cell
from .rhcr_adapted import IssuedPathAction, Point
from .joint_reconstruction import JointExecutionReconstructionEstimator, JointReconstructionEstimate


def resolve_legal_joint_moves_batch_fast(positions: np.ndarray, proposals: np.ndarray) -> np.ndarray:
    """Same ordered filter, avoiding coordinate-axis temporary reductions.

    Start occupancy includes the current agent as in the reference. Crucially,
    duplicate checks use the proposal AFTER crossing/start-occupancy reversion.
    No conflict checks are dropped or reordered.
    """
    positions = np.asarray(positions)
    proposals = np.asarray(proposals)
    if positions.ndim != 3 or positions.shape[-1] != 2 or proposals.shape != positions.shape:
        raise ValueError("positions and proposals must share shape (particles, agents, 2)")
    accepted = positions.copy()
    for i in range(positions.shape[1]):
        current = positions[:, i, :]
        proposal = proposals[:, i, :].copy()
        if i:
            starts = positions[:, :i, :]
            ends = accepted[:, :i, :]
            delta = proposal - current
            prior_delta = ends - starts
            diagonal = (np.abs(delta[:, 0]) == 1) & (np.abs(delta[:, 1]) == 1)
            prior_diagonal = (np.abs(prior_delta[:, :, 0]) == 1) & (np.abs(prior_delta[:, :, 1]) == 1)
            midpoint = current + proposal
            prior_midpoint = starts + ends
            same_midpoint = (midpoint[:, 0, None] == prior_midpoint[:, :, 0]) & (midpoint[:, 1, None] == prior_midpoint[:, :, 1])
            crossing = diagonal & np.any(prior_diagonal & same_midpoint, axis=1)
            proposal[crossing] = current[crossing]
        occupied = np.any((proposal[:, 0, None] == positions[:, :, 0]) & (proposal[:, 1, None] == positions[:, :, 1]), axis=1)
        proposal[occupied] = current[occupied]
        if i:
            duplicate = np.any((proposal[:, 0, None] == accepted[:, :i, 0]) & (proposal[:, 1, None] == accepted[:, :i, 1]), axis=1)
            proposal[duplicate] = current[duplicate]
        accepted[:, i, :] = proposal
    return accepted


class FastJointExecutionReconstructionEstimator(JointExecutionReconstructionEstimator):
    """V1 estimator semantics with exact bounded memoization and array kernels."""
    # Keep the same method identity: this is an implementation, not a model.
    def __init__(self, buildings: np.ndarray, deviation_probability: float = .25,
                 particle_count: int = 256, seed: int = 9173):
        super().__init__(buildings, deviation_probability, particle_count, seed)
        self._hold_cache = OrderedDict()

    def _holds(self, timestamp: int, agent: int) -> np.ndarray:
        key = self.seed, self.particle_count, self.deviation_probability, timestamp, agent
        if key in self._hold_cache:
            self._hold_cache.move_to_end(key)
            return self._hold_cache[key]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, timestamp, agent]))
        uniforms = (np.arange(self.particle_count) + rng.random(self.particle_count)) / self.particle_count
        rng.shuffle(uniforms)
        mask = uniforms < self.deviation_probability
        mask.flags.writeable = False
        self._hold_cache[key] = mask
        if len(self._hold_cache) > 4096:
            self._hold_cache.popitem(last=False)
        return mask

    def estimate(self, messages: list[StateMessage], paths_by_agent: list[dict[int, list[Point]]],
                 actions_by_agent: list[list[IssuedPathAction]], current_step: int) -> JointReconstructionEstimate:
        n = len(messages)
        if n == 0 or len(paths_by_agent) != n or len(actions_by_agent) != n:
            raise ValueError("one nonempty telemetry/path/action collection is required per agent")
        if {m.drone_id for m in messages} != set(range(n)):
            raise ValueError("messages must contain each consecutive agent ID exactly once")
        messages = sorted(messages, key=lambda m: m.drone_id)
        timestamps = {m.timestamp for m in messages}
        if len(timestamps) != 1:
            raise ValueError("mixed-timestamp asynchronous anchors are unsupported; retain a causal joint history instead")
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
        particles = np.tile(np.asarray(initial, dtype=np.int64), (self.particle_count, 1, 1))
        for timestamp in range(anchor + 1, current_step + 1):
            references, explicit_holds, random_holds = [], [], []
            for i in range(n):
                action = ledgers[i].get(timestamp)
                if action is None:
                    raise ValueError("complete post-anchor command history is required for every agent")
                path = paths_by_agent[i].get(action.path_id)
                if path is None or not 0 <= action.path_index < len(path):
                    raise ValueError("issued command refers to an unavailable waypoint")
                references.append(path[action.path_index])
                explicit_holds.append(action.explicit_hover)
                random_holds.append(self._holds(timestamp, i))
            proposals = particles.copy()
            for i in range(n):
                if explicit_holds[i]:
                    continue
                moving = ~random_holds[i]
                if not np.any(moving):
                    continue
                # Evaluate the tracker once per distinct starting cell, rather
                # than once for every particle sharing that cell.
                cells = particles[moving, i, :]
                height = self.buildings.shape[0]
                unique, inverse = np.unique(cells[:, 0] * height + cells[:, 1], return_inverse=True)
                projected = np.asarray([self._project((int(c // height), int(c % height)), references[i]) for c in unique])
                proposals[moving, i, :] = projected[inverse]
            particles = resolve_legal_joint_moves_batch_fast(particles, proposals)
        particle_states = [tuple((int(p[0]), int(p[1])) for p in state) for state in particles]
        joint_counts = Counter(particle_states)
        joint_map = max(joint_counts, key=lambda state: (joint_counts[state], state))
        distributions, marginal_positions = [], []
        for i in range(n):
            counts = Counter(state[i] for state in particle_states)
            distributions.append(tuple(sorted((p, count / self.particle_count) for p, count in counts.items())))
            marginal_positions.append(max(counts, key=lambda p: (counts[p], p[1], p[0])))
        return JointReconstructionEstimate(joint_map, tuple(marginal_positions), tuple(distributions),
                                           len(joint_counts), self.particle_count)
