from __future__ import annotations

from math import isclose

import pytest

from rcsr_rhcr import (
    IssuedCommand,
    MotionGraph,
    ReachabilityConstrainedStateReconstructor,
    StateMessage,
)


ANCHOR = (0, 0)
EAST = (1, 0)
NORTH = (0, 1)


def small_graph(
    vertex_order: tuple[tuple[int, int], ...] = (ANCHOR, EAST, NORTH),
) -> MotionGraph:
    return MotionGraph.from_undirected_edges(
        vertices=vertex_order,
        edges=((ANCHOR, EAST), (ANCHOR, NORTH)),
    )


def reconstruct(
    ledger: tuple[IssuedCommand, ...],
    *,
    message: StateMessage = StateMessage(0, ANCHOR),
    current_time: int = 10,
    deviation_probability: float = 0.25,
    graph: MotionGraph | None = None,
):
    return ReachabilityConstrainedStateReconstructor(
        deviation_probability
    ).reconstruct(
        message,
        ledger,
        small_graph() if graph is None else graph,
        current_time,
    )


def test_no_post_message_command_uses_delivered_position() -> None:
    result = reconstruct(())
    assert result.map_position == ANCHOR
    assert result.distribution == ((ANCHOR, 1.0),)
    assert result.audit.replayed_commands == 0


def test_legal_move_splits_probability_by_frozen_deviation_rate() -> None:
    result = reconstruct((IssuedCommand(1, EAST),))
    assert result.map_position == EAST
    assert result.as_dict() == {ANCHOR: 0.25, EAST: 0.75}
    assert result.audit.legal_source_updates == 1


def test_unreachable_cross_plan_reference_cannot_teleport_mass() -> None:
    graph = MotionGraph.from_undirected_edges(
        vertices=(ANCHOR, EAST, (4, 4)),
        edges=((ANCHOR, EAST),),
    )
    result = reconstruct(
        (IssuedCommand(1, (4, 4), plan_id="replacement-plan"),),
        graph=graph,
    )
    assert result.map_position == ANCHOR
    assert result.distribution == ((ANCHOR, 1.0),)
    assert result.audit.blocked_source_mass == 1.0
    assert result.audit.rejected_advance_mass == 0.75


def test_explicit_hover_leaves_belief_unchanged() -> None:
    result = reconstruct((IssuedCommand(1, EAST, action="hover"),))
    assert result.distribution == ((ANCHOR, 1.0),)
    assert result.audit.hover_commands == 1
    assert result.audit.move_commands == 0


def test_cross_plan_replay_aggregates_in_world_coordinates() -> None:
    result = reconstruct(
        (
            IssuedCommand(1, EAST, plan_id="plan-1", path_index=1),
            IssuedCommand(2, NORTH, plan_id="plan-2", path_index=1),
        )
    )
    assert result.map_position == EAST
    assert result.as_dict() == {
        ANCHOR: 0.0625,
        EAST: 0.75,
        NORTH: 0.1875,
    }
    assert result.audit.blocked_source_mass == 0.75


def test_only_causally_available_commands_are_replayed_in_time_order() -> None:
    message = StateMessage(timestamp=2, position=ANCHOR)
    result = reconstruct(
        (
            IssuedCommand(4, NORTH),
            IssuedCommand(1, EAST),
            IssuedCommand(3, EAST),
            IssuedCommand(11, NORTH),
        ),
        message=message,
        current_time=4,
    )
    assert result.audit.replayed_commands == 2
    assert result.map_position == EAST
    assert result.as_dict() == {
        ANCHOR: 0.0625,
        EAST: 0.75,
        NORTH: 0.1875,
    }


def test_probability_mass_is_conserved_over_long_replay() -> None:
    ledger = tuple(
        IssuedCommand(index + 1, EAST if index % 2 == 0 else ANCHOR)
        for index in range(100)
    )
    result = reconstruct(ledger, current_time=100)
    assert isclose(result.probability_mass, 1.0, abs_tol=1e-12)


def test_map_ties_follow_declared_vertex_order() -> None:
    graph = small_graph(vertex_order=(EAST, ANCHOR, NORTH))
    result = reconstruct(
        (IssuedCommand(1, EAST),),
        deviation_probability=0.5,
        graph=graph,
    )
    assert result.as_dict() == {EAST: 0.5, ANCHOR: 0.5}
    assert result.map_position == EAST


def test_message_and_command_references_must_be_graph_vertices() -> None:
    with pytest.raises(ValueError, match="message position"):
        reconstruct((), message=StateMessage(0, (9, 9)))
    with pytest.raises(ValueError, match="command reference"):
        reconstruct((IssuedCommand(1, (9, 9)),))


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="deviation_probability"):
        ReachabilityConstrainedStateReconstructor(1.01)
    with pytest.raises(ValueError, match="unique"):
        MotionGraph((ANCHOR, ANCHOR), frozenset())
    with pytest.raises(ValueError, match="current_time"):
        reconstruct((), message=StateMessage(2, ANCHOR), current_time=1)
