"""Common-timestamp observations over the unchanged asynchronous link.

The link still delays, drops, and reorders individual UAV packets. Only packets
that have actually arrived enter the buffer. The controller receives the newest
timestamp for which every UAV's packet has arrived; it may skip incomplete
timestamps. All compared controllers must use this same observation protocol.

This adapter preserves the synchronous-anchor assumption of V3's inference.
It does not implement latest-per-UAV asynchronous inference. Synchronization
can increase observation age, especially under loss or larger fleets.
An optional validator accepts complete snapshots using their reported fields;
rejected snapshots retain the previous accepted anchor and add observation age.
"""
from __future__ import annotations

from statistics import mean

from urban_lifelong_planning.async_network_channel import AsyncNetworkChannel


class CommonTimestampAsyncChannel(AsyncNetworkChannel):
    """Original asynchronous transport with a delivered common-snapshot buffer."""

    feedback_protocol = "async_link_common_timestamp_snapshot"
    synchronized_anchors = True

    def __init__(self, drone_count, trace, snapshot_validator=None):
        super().__init__(drone_count, trace)
        if self.drone_count < 1:
            raise ValueError("drone_count must be positive")
        self._arrived_by_timestamp = {}
        self._common_states = None
        self._common_timestamp = -1
        self._visible_clock = 0
        self._snapshot_updates = 0
        self._snapshot_reads = {}
        self._snapshot_validator = snapshot_validator
        self._rejected_snapshots = {}

    def bootstrap(self, messages):
        # The caller supplies the same public launch observations as V3.
        # No environment, current position, or future packet is read here.
        if (len(messages) != self.drone_count
                or {m.drone_id for m in messages} != set(range(self.drone_count))
                or any(m.timestamp != 0 for m in messages)):
            raise ValueError("bootstrap requires one public timestamp-zero message per UAV")
        if self._common_states is not None:
            raise ValueError("channel is already bootstrapped")
        ordered = sorted(messages, key=lambda message: message.drone_id)
        if self._snapshot_validator is not None and not self._snapshot_validator(ordered):
            raise ValueError("public launch snapshot is inadmissible")
        super().bootstrap(messages)

    def _accept(self, messages, visible_step=None):
        if visible_step is None:
            raise ValueError("packet arrival requires an explicit visibility step")
        if visible_step < self._visible_clock:
            raise ValueError("packet visibility time cannot go backwards")
        for message in messages:
            if not 0 <= message.drone_id < self.drone_count:
                raise ValueError("invalid UAV id")
            if not 0 <= message.timestamp <= visible_step:
                raise AssertionError("future state leaked into delivered packet buffer")
        self._visible_clock = visible_step
        # Preserve original latest-per-UAV network and out-of-order diagnostics.
        # Buffer EVERY arrived packet, even if superseded for that UAV.
        super()._accept(messages, visible_step=visible_step)
        for message in messages:
            if (message.timestamp <= self._common_timestamp
                    or message.timestamp in self._rejected_snapshots):
                continue
            snapshot = self._arrived_by_timestamp.setdefault(message.timestamp, {})
            existing = snapshot.get(message.drone_id)
            if existing is not None and existing != message:
                raise ValueError("conflicting packets for the same UAV and timestamp")
            snapshot[message.drone_id] = message
        complete = [timestamp for timestamp, snapshot in self._arrived_by_timestamp.items()
                    if len(snapshot) == self.drone_count]
        accepted = []
        for timestamp in sorted(complete):
            snapshot = self._arrived_by_timestamp[timestamp]
            states = [snapshot[i] for i in range(self.drone_count)]
            if self._snapshot_validator is None or self._snapshot_validator(states):
                accepted.append(timestamp)
            else:
                self._rejected_snapshots[timestamp] = visible_step
                del self._arrived_by_timestamp[timestamp]
        if accepted:
            timestamp = max(accepted)
            snapshot = self._arrived_by_timestamp[timestamp]
            self._common_states = tuple(snapshot[i] for i in range(self.drone_count))
            self._common_timestamp = timestamp
            self._snapshot_updates += int(timestamp > 0)
            # Older incomplete snapshots can never improve the chosen anchor.
            self._arrived_by_timestamp = {
                t: packets for t, packets in self._arrived_by_timestamp.items()
                if t > timestamp
            }

    def deliver(self, step):
        if step < self._visible_clock:
            raise ValueError("delivery time cannot go backwards")
        super().deliver(step)
        self._visible_clock = step

    def _check_read(self, step):
        if self._common_states is None:
            raise RuntimeError("planner has not received an initial common snapshot")
        if step < self._visible_clock or self._common_timestamp > step:
            raise AssertionError("delivered information cannot be read before its visibility time")

    def planner_states(self, step):
        self._check_read(step)
        self._snapshot_reads[step] = self.snapshot_diagnostics(step)
        return list(self._common_states)

    def snapshot_diagnostics(self, step):
        """Age cost compared with latest-per-UAV messages already delivered.

        Diagnostics inspect delivered packets only. They never supply the newer
        per-UAV positions to the controller or consult the link's future trace.
        """
        self._check_read(step)
        latest = [self._delivered[i].timestamp for i in range(self.drone_count)]
        extra = [timestamp - self._common_timestamp for timestamp in latest]
        return {
            "common_snapshot_timestamp": self._common_timestamp,
            "common_snapshot_aoi": step - self._common_timestamp,
            "latest_per_uav_mean_aoi": mean(step - timestamp for timestamp in latest),
            "synchronization_extra_mean_aoi": mean(extra),
            "synchronization_extra_max_aoi": max(extra),
            "buffered_incomplete_snapshot_count": len(self._arrived_by_timestamp),
            "rejected_complete_snapshot_count": len(self._rejected_snapshots),
        }

    def communication_summary(self):
        result = super().communication_summary()
        result.update(
            feedback_protocol=self.feedback_protocol,
            common_snapshot_updates=self._snapshot_updates,
            snapshot_observation_count=len(self._snapshot_reads),
            snapshot_summary_weighting="one observation per distinct planner query step",
            snapshot_validator_enabled=self._snapshot_validator is not None,
            rejected_complete_snapshot_count=len(self._rejected_snapshots),
            rejected_complete_snapshots=[
                {"timestamp": timestamp, "visible_step": visible_step}
                for timestamp, visible_step in self._rejected_snapshots.items()],
        )
        rows = list(self._snapshot_reads.values())
        for field in ("common_snapshot_aoi", "latest_per_uav_mean_aoi",
                      "synchronization_extra_mean_aoi", "synchronization_extra_max_aoi"):
            result["mean_" + field] = mean(row[field] for row in rows) if rows else None
            result["max_" + field] = max(row[field] for row in rows) if rows else None
        return result
