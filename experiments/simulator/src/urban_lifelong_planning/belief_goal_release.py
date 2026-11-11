"""Conservative belief-conditioned release of terminal goal padding.

Separation guarantees are conditional on the supplied positive support, not
a promise that approximate or particle beliefs cover every physical outcome.
The shared execution safety filter remains active.

No environment, random tape or execution-outcome input. Keeps the original
reference; it only distinguishes physical hold from tracking that reference.
"""
from math import hypot

def goal_release_mask(beliefs, goals, paths, assigned, execution_horizon):
    result=[False]*len(paths)
    if len(beliefs) != len(paths) or len(goals) != len(paths):
        raise ValueError("one belief and goal are required per path")
    if execution_horizon < 1:
        raise ValueError("execution horizon must be positive")
    # No envelope claim is justified if any agent lacks positive support.
    if any(not any(w > 0 for w in belief.values()) for belief in beliefs):
        return result
    for i,path in enumerate(paths):
        if i not in assigned or not path or any(p!=path[0] for p in path):
            continue
        goal=goals[i]
        if hypot(path[0][0]-goal[0],path[0][1]-goal[1])>1.:
            continue
        support=[p for p,w in beliefs[i].items() if w>0]
        if not any(hypot(p[0]-goal[0],p[1]-goal[1])>1. for p in support):
            continue
        # Chebyshev distance can shrink by at most two per execution step.
        # Strict > 2H guarantees no vertex/start/segment conflict with any
        # other agent throughout this open-loop prefix, for all support and
        # legal motion outcomes. Conservative marginal supports are sufficient.
        if any(max(abs(p[0]-q[0]),abs(p[1]-q[1])) <= 2*execution_horizon
               for j,b in enumerate(beliefs) if j!=i for p in support
               for q,w in b.items() if w>0):
            continue
        result[i]=True
    return result
