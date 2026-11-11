"""Causal joint particle replay for synchronized delayed telemetry.

This is a separate experimental estimator, not the paper's original RCSR.
It reproduces the discrete deployed tracker and safety filter, not continuous
vehicle dynamics. Equal-weight stratified Monte Carlo particles approximate
joint uncertainty; no true execution states or simulator random tape enter.
Mixed-age asynchronous anchors are explicitly unsupported: a stateless filter
cannot reconstruct older joint states from newer marginal observations.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import numpy as np

from .delayed_state_channel import StateMessage
from .motion_graph import legal_cell, legal_transition
from .rhcr_adapted import IssuedPathAction, Point, _diagonals_cross, reference_projection


@dataclass(frozen=True)
class JointReconstructionEstimate:
    positions: tuple[Point, ...]
    marginal_positions: tuple[Point, ...]
    distributions: tuple[tuple[tuple[Point, float], ...], ...]
    joint_support_size: int
    particle_count: int


def resolve_joint_moves(buildings: np.ndarray, positions: tuple[Point, ...], proposals: tuple[Point, ...]) -> tuple[Point, ...]:
    """Pure equivalent of the discrete environment's ordered safety filter."""
    if len(positions) != len(proposals):
        raise ValueError("one proposal is required per agent")
    accepted = list(positions)
    start_occupied = set(positions)
    occupied: set[Point] = set()
    for i, proposal in enumerate(proposals):
        current = positions[i]
        if not legal_transition(buildings, current, proposal):
            proposal = current
        if any(_diagonals_cross(current, proposal, positions[j], accepted[j]) for j in range(i)):
            proposal = current
        if proposal != current and proposal in start_occupied:
            proposal = current
        if proposal in occupied:
            proposal = current
        accepted[i] = proposal
        occupied.add(proposal)
    return tuple(accepted)


def resolve_legal_joint_moves_batch(positions: np.ndarray, proposals: np.ndarray) -> np.ndarray:
    """Vectorized ordered safety filtering for already graph-legal proposals.

    Arrays have shape (particles, agents, 2). Callers must construct proposals
    by a legal tracker or a hold; the scalar resolver additionally validates
    arbitrary proposals and remains the reference oracle for this fast path.
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
            prior_starts = positions[:, :i, :]
            prior_ends = accepted[:, :i, :]
            diagonal = np.all(np.abs(proposal - current) == 1, axis=1)
            prior_diagonal = np.all(np.abs(prior_ends - prior_starts) == 1, axis=2)
            same_midpoint = np.all((current + proposal)[:, None, :] == prior_starts + prior_ends, axis=2)
            crossing = diagonal & np.any(prior_diagonal & same_midpoint, axis=1)
            proposal[crossing] = current[crossing]
        start_occupied = np.any(np.all(proposal[:, None, :] == positions, axis=2), axis=1)
        proposal[start_occupied] = current[start_occupied]
        if i:
            duplicate = np.any(np.all(proposal[:, None, :] == accepted[:, :i, :], axis=2), axis=1)
            proposal[duplicate] = current[duplicate]
        accepted[:, i, :] = proposal
    return accepted


class JointExecutionReconstructionEstimator:
    """Replay delivered joint anchors using only map and command history.

    ``positions`` is an empirical joint MAP state, so unlike independent
    marginal MAP decisions it belongs to joint support and has distinct cells.
    Stratification makes the marginal number of random holds accurate to one
    particle, with independent reproducible permutations per time/agent.
    ``seed`` is algorithm randomness, independent of every execution tape.
    """
    method_name = "Joint execution replay"
    physical_anchor_belief = True

    def __init__(self, buildings: np.ndarray, deviation_probability: float = .25,
                 particle_count: int = 256, seed: int = 9173):
        if not 0. <= deviation_probability <= 1.:
            raise ValueError("deviation_probability must lie in [0, 1]")
        if particle_count < 1 or int(particle_count) != particle_count:
            raise ValueError("particle_count must be a positive integer")
        if seed < 0 or int(seed) != seed:
            raise ValueError("seed must be a nonnegative integer")
        self.buildings = np.array(buildings, dtype=bool, copy=True)
        self.buildings.flags.writeable = False
        self.deviation_probability = float(deviation_probability)
        self.particle_count = int(particle_count)
        self.seed = int(seed)
        self._projection_cache: dict[tuple[Point, Point], Point] = {}

    def _project(self, position: Point, reference: Point) -> Point:
        key = position, reference
        if key not in self._projection_cache:
            if len(self._projection_cache) >= 65536:
                self._projection_cache.clear()
            self._projection_cache[key] = reference_projection(self.buildings, position, reference)
        return self._projection_cache[key]

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
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, timestamp, i]))
                uniforms = (np.arange(self.particle_count) + rng.random(self.particle_count)) / self.particle_count
                rng.shuffle(uniforms)
                random_holds.append(uniforms < self.deviation_probability)
            proposals = particles.copy()
            for i in range(n):
                if explicit_holds[i]:
                    continue
                moving = ~random_holds[i]
                if not np.any(moving):
                    continue
                # Evaluate the tracker once per distinct starting cell, rather
                # than once for every particle sharing that cell.
                unique, inverse = np.unique(particles[moving, i, :], axis=0, return_inverse=True)
                projected = np.asarray([self._project((int(p[0]), int(p[1])), references[i]) for p in unique])
                proposals[moving, i, :] = projected[inverse]
            particles = resolve_legal_joint_moves_batch(particles, proposals)
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
