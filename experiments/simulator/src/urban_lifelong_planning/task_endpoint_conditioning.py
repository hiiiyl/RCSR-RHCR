"""Optional marginal endpoint filtering using instantaneous public task notices.

Not a joint conditional posterior or a history-aware observation filter. Never
accepts an environment, true position, execution outcome, or random tape.
"""
from dataclasses import dataclass
from math import hypot, isfinite


@dataclass(frozen=True)
class PublicTaskNotice:
    target_id: int
    position: tuple[int, int]
    assigned_uav: int | None
    assignment_step: int | None
    completion_step: int | None


@dataclass(frozen=True)
class EndpointConstraint:
    position: tuple[int, int]
    radius: float
    inside: bool

    def allows(self, point):
        return (hypot(point[0]-self.position[0], point[1]-self.position[1]) <= self.radius) == self.inside


def endpoint_constraints(active, completed, assignments, current_step, fleet, radius=1.0, *, eligible=True):
    """Derive only current-endpoint facts; uncertainty disables evidence.

    eligible certifies the audited successful discrete reserved-step semantics,
    not battery telemetry or a guessed service eligibility condition.
    """
    result = [[] for _ in range(fleet)]
    if not eligible or current_step <= 0:
        return result
    if radius < 0 or not isfinite(radius):
        raise ValueError('invalid service radius')
    for task in active:
        i = task.assigned_uav
        if (i is not None and 0 <= i < fleet and assignments.get(i) == task.target_id
                and task.completion_step is None and task.assignment_step is not None
                and 0 <= task.assignment_step < current_step):
            result[i].append(EndpointConstraint(task.position, radius, False))
    for task in completed:
        i = task.assigned_uav
        if (i is not None and 0 <= i < fleet and task.completion_step == current_step
                and task.assignment_step is not None and 0 <= task.assignment_step < current_step):
            result[i].append(EndpointConstraint(task.position, radius, True))
    return result


def filter_marginal_endpoint(belief, constraints):
    """Return a new marginal, retained mass, and explicit contradiction flag.

    Zero retained support fails open to the input distribution for control only.
    It is always reported as a contradiction, never a valid conditioning result.
    """
    kept = {p: mass for p, mass in belief.items() if all(c.allows(p) for c in constraints)}
    total = sum(kept.values())
    if not constraints:
        return dict(belief), sum(belief.values()), False
    if total <= 0:
        return dict(belief), total, True
    return {p: mass / total for p, mass in kept.items()}, total, False
