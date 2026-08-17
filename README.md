# RCSR-RHCR Core

Minimal reference implementation of **Reachability-Constrained State
Reconstruction for Rolling-Horizon Multi-UAV Planning under Delayed
Feedback**.

RCSR reconstructs the spatial state used to start a new rolling-horizon
planning call. It anchors a probability distribution at the latest delivered
historical position and replays only controller-issued commands that are
causally available. Probability can advance to a reference waypoint only
through an edge of the planner's legal motion graph.

## Contents

This repository contains:

- the graph-constrained state-reconstruction kernel;
- a small synthetic example;
- tests of the method's principal invariants; and
- a concise method description.

The synthetic example requires no external dataset. The full multi-UAV
simulation and experimental data are outside the scope of this core reference
implementation.

## Requirements

- Python 3.10 or later
- no third-party runtime dependencies
- `pytest` only for running the tests

## Install and verify

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pytest -q
```

Run the minimal example:

```bash
python examples/minimal_reconstruction.py
```

Expected MAP anchor:

```text
(1, 0)
```

## Minimal use

```python
from rcsr_rhcr import (
    IssuedCommand,
    MotionGraph,
    ReachabilityConstrainedStateReconstructor,
    StateMessage,
)

graph = MotionGraph.from_undirected_edges(
    vertices=((0, 0), (1, 0), (0, 1)),
    edges=(((0, 0), (1, 0)), ((0, 0), (0, 1))),
)
message = StateMessage(timestamp=0, position=(0, 0))
ledger = (IssuedCommand(timestamp=1, reference=(1, 0)),)

result = ReachabilityConstrainedStateReconstructor(
    deviation_probability=0.25
).reconstruct(message, ledger, graph, current_time=1)

print(result.map_position)
print(result.distribution)
```

The graph uses directed edges. `from_undirected_edges` inserts both
directions. Vertex tuple order is the deterministic MAP tie-break order.

## Integration contract

RCSR changes only the planning-start interface. A host rolling-horizon system
must provide:

1. the latest delivered timestamped state;
2. the controller's issued-command ledger;
3. the legal motion graph used by the planner; and
4. the current planning time.

The implementation must not read the live vehicle state, future telemetry, or
executed outcomes when reconstructing the planning anchor. Task allocation,
conflict handling, path search, and execution remain outside this package.

## Research-use warning

This is a research reference artifact, not flight-control software. It has not
been certified for safety-critical or operational UAV deployment.

## Citation

If this code supports academic work, cite the associated paper:

> Yonghai Liu, "Reachability-Constrained State Reconstruction for
> Rolling-Horizon Multi-UAV Planning under Delayed Feedback."

Repository: <https://github.com/hiiiyl/RCSR-RHCR>

Publication metadata will be added to `CITATION.cff` after the paper receives
its final bibliographic record.

## License

This is source-available research code, not open-source software. Copyright is
retained; the narrow permission is limited to running unmodified copies for
non-commercial academic inspection and verification. See [LICENSE](LICENSE).
