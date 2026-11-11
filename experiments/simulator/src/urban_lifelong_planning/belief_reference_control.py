"""Causal marginal forward references for support-separated agents only.

Nominal paths remain untouched. References are tracker inputs, not executed
edges. No guarantee is made outside finite posterior support. Coupled agents
retain original control, avoiding a fabricated independent joint posterior.
"""
from collections import defaultdict
import numpy as np
from math import hypot
from .motion_graph import legal_cell, legal_transition
from .rhcr_adapted import reference_projection, _goal_distance_map


class SeparatedForwardReferenceControl:
    name = 'support_separated_forward_v1'

    def __init__(self, buildings, hold_probability=.25):
        self.buildings = buildings
        self.hold_probability = hold_probability
        self._distances = {}
        self._projections = {}

    def distances(self, goal):
        if goal not in self._distances:
            distances = _goal_distance_map(self.buildings.shape,
                np.asarray(self.buildings,dtype=np.bool_).tobytes(),goal,1.)
            ys,xs = np.where(np.isfinite(distances))
            self._distances[goal] = {(int(x),int(y)):int(distances[y,x]) for y,x in zip(ys,xs)}
        return self._distances[goal]

    def predict(self, belief, reference, hold=False):
        if hold:
            return belief.copy()
        updated = defaultdict(float)
        for point,mass in belief.items():
            key = point,reference
            if key not in self._projections:
                self._projections[key] = reference_projection(self.buildings,point,reference)
            updated[point] += mass*self.hold_probability
            updated[self._projections[key]] += mass*(1.-self.hold_probability)
        return dict(updated)

    def choose(self, beliefs, goals, paths, assigned, release, horizon):
        n = len(paths)
        indices = [[j]*n for j in range(1,horizon+1)]
        holds = [[paths[i][j] == paths[i][j-1] and not release[i] for i in range(n)] for j in range(1,horizon+1)]
        supports = [{p for p,w in b.items() if w>0} for b in beliefs]
        eligible = [i in assigned and len(supports[i]) > 1 and all(supports) and not any(
            max(abs(p[0]-q[0]),abs(p[1]-q[1])) <= 2*horizon
            for k in range(n) if k!=i for p in supports[i] for q in supports[k]) for i in range(n)]
        gains, changed = 0.,0
        for i in range(n):
            if not eligible[i]:
                continue
            distance = self.distances(goals[i])
            belief = beliefs[i].copy()
            for j in range(1,horizon+1):
                if holds[j-1][i]:
                    continue
                baseline = self.predict(belief, paths[i][j])
                if any(p not in distance for p in baseline):
                    belief = baseline
                    continue
                best_score = sum(w*distance[p] for p,w in baseline.items())
                old_score = best_score
                best = baseline
                # Never cross a nominal wait, including terminal padding.
                for k in range(j+1,len(paths[i])):
                    if paths[i][k] == paths[i][k-1]:
                        break
                    next_belief = self.predict(belief, paths[i][k])
                    if any(p not in distance for p in next_belief):
                        continue
                    score = sum(w*distance[p] for p,w in next_belief.items())
                    if score < best_score - 1e-12:
                        best_score, best = score,next_belief
                        indices[j-1][i] = k
                gains += old_score-best_score
                changed += indices[j-1][i] != j
                belief = best
        return indices, holds, {'forward_eligible_agents':sum(eligible), 'forward_changed_references':changed,
            'forward_expected_distance_gain':gains, 'forward_changed_agents_supported_intervention_risk':0.}
