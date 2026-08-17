# Method summary

## Information boundary

For one UAV, reconstruction uses only:

- the latest delivered message `m = (timestamp, position, metadata)`;
- the controller-issued command ledger after `m.timestamp`;
- the legal motion graph `G = (V, E)`; and
- the current planning time.

Plan identifiers and path indices are retained only as ledger metadata. The
probabilistic state is represented in world coordinates.

## Reconstruction

Let `p_d` be the fixed probability that an issued move does not advance. The
initial belief is

```text
B = {m.position: 1}
```

Commands with timestamps in `(m.timestamp, current_time]` are replayed in
temporal order. For a move command with reference waypoint `r`, every
hypothesis `(x, q)` is updated as follows:

```text
if (x, r) is a legal graph edge:
    next[x] += p_d * q
    next[r] += (1 - p_d) * q
else:
    next[x] += q
```

An explicit hover leaves the belief unchanged. After all eligible commands
are replayed, the MAP world position is selected. Equal-probability positions
are resolved by the graph's declared vertex order.

## Required invariants

- probability mass remains one within floating-point tolerance;
- support remains on legal graph vertices;
- an unreachable reference receives no probability mass;
- a hover creates no fictitious progress;
- a newly delivered message replaces the historical anchor;
- future or already-acknowledged commands are not replayed; and
- no live vehicle state is read by reconstruction.

## Host-system integration

The returned `map_position` can replace the stale message position only at the
planning-start boundary. The task allocator, space-time path search, conflict
rules, and executor are intentionally outside this public core.
