"""Run a small cross-plan RCSR reconstruction example."""

from rcsr_rhcr import (
    IssuedCommand,
    MotionGraph,
    ReachabilityConstrainedStateReconstructor,
    StateMessage,
)


def main() -> None:
    anchor = (0, 0)
    east = (1, 0)
    north = (0, 1)
    graph = MotionGraph.from_undirected_edges(
        vertices=(anchor, east, north),
        edges=((anchor, east), (anchor, north)),
    )
    message = StateMessage(
        timestamp=0,
        position=anchor,
        plan_id="plan-0",
        path_index=0,
    )
    ledger = (
        IssuedCommand(
            timestamp=1,
            reference=east,
            plan_id="plan-1",
            path_index=1,
        ),
        IssuedCommand(
            timestamp=2,
            reference=north,
            plan_id="plan-2",
            path_index=1,
        ),
    )

    result = ReachabilityConstrainedStateReconstructor(0.25).reconstruct(
        message=message,
        ledger=ledger,
        graph=graph,
        current_time=2,
    )

    print(f"MAP planning anchor: {result.map_position}")
    print("Spatial distribution:")
    for point, probability in result.distribution:
        print(f"  {point}: {probability:.4f}")
    print(f"Rejected advance mass: {result.audit.rejected_advance_mass:.4f}")


if __name__ == "__main__":
    main()
