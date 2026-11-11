"""Independent coordinate and deployed-tracker compatibility contracts.

A physical posterior need not replay the deployed reference_projection rule.
Compatibility here authorizes the shared forward controller, never rewrites an
estimator or expands the goal-release contract. Unknown providers fail closed.
"""
from dataclasses import dataclass
from .rhcr_adapted import (ExecutionModelConsistentEstimator,
    DeterministicReplayEstimator, ExecutionProgressStartEstimator,
    IndexOnlyProgressStartEstimator)
from .joint_reconstruction import JointExecutionReconstructionEstimator


@dataclass(frozen=True)
class ReferenceTrackerCapability:
    physical_coordinates: bool
    compatible_forward: bool
    model: str
    reason: str | None = None


def reference_tracker_capability(provider):
    if provider is None:
        return ReferenceTrackerCapability(True, True, 'singleton_point_adapter')
    physical = bool(getattr(provider, 'physical_anchor_belief', False))
    if isinstance(provider, (ExecutionModelConsistentEstimator,
                             JointExecutionReconstructionEstimator)):
        return ReferenceTrackerCapability(physical, physical, 'reference_projection',
            None if physical else 'nonphysical_coordinate_representation')
    if isinstance(provider, DeterministicReplayEstimator):
        # Always singleton, so the frozen controller retains nominal references.
        return ReferenceTrackerCapability(physical, physical, 'reference_projection_singleton',
            None if physical else 'nonphysical_coordinate_representation')
    if isinstance(provider, ExecutionProgressStartEstimator):
        return ReferenceTrackerCapability(physical, False, 'adjacent_reference_only',
            'estimator does not project distant references')
    if isinstance(provider, IndexOnlyProgressStartEstimator):
        return ReferenceTrackerCapability(physical, False, 'nominal_path_index',
            'belief is not a physical-coordinate tracker posterior')
    return ReferenceTrackerCapability(physical, False,
        getattr(provider, 'reference_tracker_model', 'unverified_tracker'),
        'estimator has no verified deployed reference-projection transition')
