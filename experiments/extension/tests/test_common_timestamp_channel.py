"""Focused causal-buffer checks; no episode or formal panel is executed."""
from pathlib import Path
import sys

EXTENSION = Path(__file__).resolve().parents[1]
V3 = EXTENSION.parents[1]
sys.path.insert(0, str(EXTENSION.parent / "simulator" / "src"))
sys.path.insert(0, str(EXTENSION))

import pytest

from common_timestamp_channel import CommonTimestampAsyncChannel
from urban_lifelong_planning.delayed_state_channel import DelayedStateChannel, StateMessage


def messages(step, count=2):
    return [StateMessage(i, step, (step, i), 1500.0 - step, None,
                         step % 2, path_id=step // 2) for i in range(count)]


def trace(steps=15, count=2, delay=0, changes=None):
    data = {str(t): {str(i): [delay, False] for i in range(count)}
            for t in range(steps + 1)}
    for (t, i), value in (changes or {}).items():
        data[str(t)][str(i)] = value
    return {"messages": data}


def test_reordered_old_packet_can_complete_newest_common_anchor():
    channel = CommonTimestampAsyncChannel(2, trace(changes={
        (1, 0): [5, False], (2, 1): [5, False]}))
    channel.bootstrap(messages(0))
    channel.send(messages(1))
    channel.deliver(1)
    channel.send(messages(2))
    channel.deliver(2)
    assert [m.timestamp for m in channel.planner_states(2)] == [0, 0]
    assert channel.snapshot_diagnostics(2)["synchronization_extra_mean_aoi"] == 1.5
    channel.deliver(6)
    # UAV 0's timestamp-1 packet arrives after its timestamp-2 packet.
    assert channel.planner_states(6) == messages(1)
    channel.deliver(7)
    assert channel.planner_states(7) == messages(2)
    assert channel.communication_summary()["actual_out_of_order_count"] == 1


def test_lost_packet_does_not_block_later_complete_snapshot():
    channel = CommonTimestampAsyncChannel(2, trace(changes={(1, 1): [0, True]}))
    channel.bootstrap(messages(0))
    channel.send(messages(1))
    channel.deliver(1)
    assert channel.planner_states(1) == messages(0)
    channel.send(messages(2))
    channel.deliver(2)
    assert channel.planner_states(2) == messages(2)
    assert channel.snapshot_diagnostics(2)["buffered_incomplete_snapshot_count"] == 0
    assert channel.communication_summary()["total_messages_dropped"] == 1


@pytest.mark.parametrize("delay", [0, 2, 6])
def test_fixed_delay_observations_equal_original_channel(delay):
    common = CommonTimestampAsyncChannel(2, trace(delay=delay))
    original = DelayedStateChannel(2, delay)
    for channel in (common, original):
        channel.bootstrap(messages(0))
    for step in range(1, 16):
        for channel in (common, original):
            channel.send(messages(step))
            channel.deliver(step)
        assert common.planner_states(step) == original.planner_states(step)
        assert common.aoi(step) == original.aoi(step)
        assert common.snapshot_diagnostics(step)["synchronization_extra_mean_aoi"] == 0


def test_pending_packets_are_invisible_and_past_reads_rejected():
    channel = CommonTimestampAsyncChannel(2, trace(delay=4))
    channel.bootstrap(messages(0))
    channel.send(messages(1))
    channel.deliver(4)
    assert channel.planner_states(4) == messages(0)
    assert channel.snapshot_diagnostics(4)["latest_per_uav_mean_aoi"] == 4
    channel.deliver(5)
    assert channel.planner_states(5) == messages(1)
    with pytest.raises(AssertionError, match="visibility time"):
        channel.planner_states(4)


def test_never_reads_future_message_at_accept():
    channel = CommonTimestampAsyncChannel(2, trace())
    channel.bootstrap(messages(0))
    with pytest.raises(AssertionError, match="future"):
        channel._accept(messages(2), visible_step=1)
    assert channel.planner_states(0) == messages(0)


def test_bootstrap_requires_only_explicit_complete_launch_observations():
    channel = CommonTimestampAsyncChannel(2, trace())
    with pytest.raises(RuntimeError, match="initial common"):
        channel.planner_states(0)
    with pytest.raises(ValueError, match="timestamp-zero"):
        channel.bootstrap(messages(1))
    with pytest.raises(ValueError, match="per UAV"):
        channel.bootstrap(messages(0)[:1])
    channel.bootstrap(messages(0))
    assert channel.planner_states(0) == messages(0)


def test_all_later_packets_lost_retains_launch_and_reports_age():
    changes = {(t, i): [0, True] for t in range(1, 5) for i in range(2)}
    channel = CommonTimestampAsyncChannel(2, trace(changes=changes))
    channel.bootstrap(messages(0))
    for step in range(1, 5):
        channel.send(messages(step))
        channel.deliver(step)
        assert channel.planner_states(step) == messages(0)
    assert channel.aoi(4) == [4, 4]
    assert channel.communication_summary()["max_common_snapshot_aoi"] == 4


def test_repeated_reads_do_not_reweight_snapshot_diagnostics():
    channel = CommonTimestampAsyncChannel(2, trace())
    channel.bootstrap(messages(0))
    channel.planner_states(0)
    channel.planner_states(0)
    channel.aoi(0)
    assert channel.communication_summary()["snapshot_observation_count"] == 1


def test_snapshot_validator_retains_last_accepted_anchor_then_recovers():
    from dataclasses import replace
    distinct = lambda states: len({m.position for m in states}) == len(states)
    channel = CommonTimestampAsyncChannel(2, trace(), snapshot_validator=distinct)
    channel.bootstrap(messages(0))
    collision = messages(1)
    collision[1] = replace(collision[1], position=collision[0].position)
    channel.send(collision)
    channel.deliver(1)
    assert channel.planner_states(1) == messages(0)
    assert channel.snapshot_diagnostics(1)["rejected_complete_snapshot_count"] == 1
    assert channel.snapshot_diagnostics(1)["synchronization_extra_mean_aoi"] == 1
    channel.send(messages(2))
    channel.deliver(2)
    assert channel.planner_states(2) == messages(2)
    assert channel.communication_summary()["rejected_complete_snapshots"] == [
        {"timestamp": 1, "visible_step": 1}]


def test_snapshot_validator_rejects_inadmissible_launch():
    channel = CommonTimestampAsyncChannel(2, trace(), snapshot_validator=lambda states: False)
    with pytest.raises(ValueError, match="launch snapshot"):
        channel.bootstrap(messages(0))
    with pytest.raises(RuntimeError, match="initial common"):
        channel.planner_states(0)
