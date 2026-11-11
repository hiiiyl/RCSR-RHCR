"""Opt-in implementation acceleration of V2 sampled components only.

Exact recursion, component boundaries, validation, diagnostics, RNG cache, and
floating-point aggregation order are inherited unchanged. This is not a new
posterior model. No global agent renumbering is performed during sampling.
"""
from collections import defaultdict
import numpy as np
from .component_reconstruction import ComponentExecutionReconstructionEstimator
from .fast_joint_reconstruction import resolve_legal_joint_moves_batch_fast


class FastComponentExecutionReconstructionEstimator(ComponentExecutionReconstructionEstimator):
    def _sample(self, ids, initial, commands, anchor):
        particles = np.tile(np.asarray([initial[i] for i in ids], dtype=np.int64), (self.particle_count, 1, 1))
        height = self.buildings.shape[0]
        for timestamp, frame in enumerate(commands, anchor + 1):
            proposals = particles.copy()
            for k, i in enumerate(ids):
                reference, hover = frame[i]
                if hover:
                    continue
                # i is the ORIGINAL global agent ID; k only indexes the array.
                moving = ~self._holds(timestamp, i)
                if not np.any(moving):
                    continue
                cells = particles[moving, k, :]
                unique, inverse = np.unique(cells[:, 0] * height + cells[:, 1], return_inverse=True)
                projected = np.asarray([self._project((int(cell // height), int(cell % height)), reference) for cell in unique])
                proposals[moving, k, :] = projected[inverse]
            particles = resolve_legal_joint_moves_batch_fast(particles, proposals)
        states = defaultdict(float)
        # Preserve V2's particle iteration, insertion, and FP addition order.
        for row in particles:
            states[tuple(tuple(map(int, pos)) for pos in row)] += 1. / self.particle_count
        return states
