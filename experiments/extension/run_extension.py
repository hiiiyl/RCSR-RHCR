"""Isolated-worker interface adapters around the unchanged V3 episode loop.

The caller must use a fresh process for each episode. Factory replacement is
process-local and is not safe for concurrent threads. The original V3 spec
continues to describe synchronized inference input; the returned summary also
identifies the actual transport condition and observation protocol.
"""
from __future__ import annotations

from .common_timestamp_channel import CommonTimestampAsyncChannel

_ACTIVE = False


class _RecordedCommonTimestampChannel(CommonTimestampAsyncChannel):
    def __init__(self, drone_count, trace, snapshot_validator=None):
        super().__init__(drone_count, trace, snapshot_validator=snapshot_validator)
        self.cycle_observations = {}

    def planner_states(self, step):
        states = super().planner_states(step)
        timestamps = {message.timestamp for message in states}
        if len(timestamps) != 1:
            raise AssertionError("common-snapshot adapter supplied mixed anchors")
        self.cycle_observations[step] = {
            **self.snapshot_diagnostics(step),
            "observation_timestamps_synchronized": True,
        }
        return states


def run_extension_episode(spec, tape_path, condition="moderate", continuous=False):
    """Run V3 with common delivered anchors and optional continuous execution.

    The spec is passed to V3 unchanged and must use condition='fixed'.
    The condition argument selects the external link for all compared methods:
    fixed = zero delay/loss; moderate = delays 0..6, loss .10;
    severe = delays 0..10, loss .20. The spec.delay field remains unchanged.
    Initial timestamp-zero observations retain V3's public launch convention.
    """
    if condition not in ("fixed", "moderate", "severe"):
        raise ValueError("link condition must be fixed, moderate, or severe")
    if spec.condition != "fixed" or spec.experiment == "continuous":
        raise ValueError("V3 inference spec must retain its synchronized discrete input")
    global _ACTIVE
    if _ACTIVE:
        raise RuntimeError("factory injection requires a separate, non-nested worker")

    from urban_lifelong_planning import revision_experiment as revision
    from urban_lifelong_planning.async_network_channel import generate_async_trace

    maximum, loss = {"fixed": (0, 0.), "moderate": (6, .10),
                     "severe": (10, .20)}[condition]
    trace = generate_async_trace(spec.seed, spec.fleet, spec.steps, maximum, loss, condition)
    channels = []
    captured_envs = []
    if continuous:
        from .continuous_adapter import (make_continuous_env_factory,
                                         make_graph_snapshot_validator)

    def channel_factory(drone_count, delay_steps):
        if drone_count != spec.fleet:
            raise ValueError("channel fleet does not match episode spec")
        validator = None
        if continuous:
            if len(captured_envs) != 1:
                raise AssertionError("continuous environment must precede channel construction")
            validator = make_graph_snapshot_validator(captured_envs[0].building_mask)
        channel = _RecordedCommonTimestampChannel(drone_count, trace, snapshot_validator=validator)
        if continuous:
            channel.feedback_protocol = "common_timestamp_graph_admissible_snapshot"
        channels.append(channel)
        return channel

    original_factory = revision.DelayedStateChannel
    original_env_factory = revision.FormalTapeCityEnv
    _ACTIVE = True
    try:
        revision.DelayedStateChannel = channel_factory
        if continuous:
            revision.FormalTapeCityEnv = make_continuous_env_factory(captured_envs)
        summary, tasks, planning = revision.run_episode(spec, tape_path)
    finally:
        revision.DelayedStateChannel = original_factory
        revision.FormalTapeCityEnv = original_env_factory
        _ACTIVE = False

    if len(channels) != 1:
        raise AssertionError("the episode must use exactly one common-snapshot channel")
    channel = channels[0]
    for record in planning:
        record.update(channel.cycle_observations[record["step"]])
        record["feedback_protocol"] = channel.feedback_protocol
        record["network_condition"] = condition
        if continuous:
            record["inference_state_semantics"] = "discrete_graph_surrogate"
            record["conditional_posterior_valid_scope"] = "discrete_graph_surrogate_only"
            record["physical_posterior_available"] = False
    summary.update(channel.communication_summary())
    summary.update(
        condition=condition,
        network_condition=condition,
        network_transport="fixed_zero_delay" if condition == "fixed" else "asynchronous_delay_loss",
        inference_input_condition=spec.condition,
        feedback_protocol=channel.feedback_protocol,
        async_max_delay_steps=maximum,
        async_loss_probability=loss,
        all_planning_observations_synchronized=all(
            record["observation_timestamps_synchronized"] for record in planning),
        runtime_interpretation="interface-extension run; timing follows the outer runner protocol",
        v3_episode_key=summary["episode_key"],
        v3_experiment=spec.experiment,
        execution_regime="continuous_PD_graph_surrogate" if continuous else "discrete_graph",
    )
    if continuous:
        # Set this scope after V3's history diagnostics have been summarized.
        summary.update(continuous=True, experiment="continuous",
                       inference_state_semantics="discrete_graph_surrogate",
                       conditional_posterior_valid_scope="discrete_graph_surrogate_only",
                       physical_posterior_available=False)
    summary["episode_key"] += ("__continuous_common_snapshot_" if continuous
                               else "__async_common_snapshot_") + condition
    return summary, tasks, planning
