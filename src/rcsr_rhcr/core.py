"""Reachability-constrained state reconstruction (RCSR)."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import fsum, isclose
from typing import Iterable, Literal, Mapping

Point = tuple[int, int]
Edge = tuple[Point, Point]
PlanId = int | str | None
Action = Literal["move", "hover"]


@dataclass(frozen=True)
class MotionGraph:
    """Legal motion graph with a deterministic vertex order.

    ``edges`` are directed. Use :meth:`from_undirected_edges` when the motion
    model permits traversal in both directions.
    """

    vertices: tuple[Point, ...]
    edges: frozenset[Edge]
    _rank: Mapping[Point, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.vertices:
            raise ValueError("the motion graph must contain at least one vertex")
        if len(set(self.vertices)) != len(self.vertices):
            raise ValueError("motion-graph vertices must be unique")
        for point in self.vertices:
            _validate_point(point, "vertex")

        vertex_set = set(self.vertices)
        for start, end in self.edges:
            _validate_point(start, "edge start")
            _validate_point(end, "edge end")
            if start not in vertex_set or end not in vertex_set:
                raise ValueError("every edge endpoint must be a graph vertex")

        object.__setattr__(
            self,
            "_rank",
            {point: index for index, point in enumerate(self.vertices)},
        )

    @classmethod
    def from_undirected_edges(
        cls,
        vertices: Iterable[Point],
        edges: Iterable[Edge],
    ) -> "MotionGraph":
        """Build a graph by inserting both directions of every given edge."""

        ordered_vertices = tuple(vertices)
        directed_edges: set[Edge] = set()
        for start, end in edges:
            directed_edges.add((start, end))
            directed_edges.add((end, start))
        return cls(ordered_vertices, frozenset(directed_edges))

    def contains(self, point: Point) -> bool:
        return point in self._rank

    def has_edge(self, start: Point, end: Point) -> bool:
        return (start, end) in self.edges

    def rank(self, point: Point) -> int:
        return self._rank[point]


@dataclass(frozen=True)
class StateMessage:
    """Latest delivered historical state used as the reconstruction anchor."""

    timestamp: int
    position: Point
    plan_id: PlanId = None
    path_index: int | None = None

    def __post_init__(self) -> None:
        _validate_point(self.position, "message position")


@dataclass(frozen=True)
class IssuedCommand:
    """Controller-issued command retained in the causal command ledger."""

    timestamp: int
    reference: Point
    action: Action = "move"
    plan_id: PlanId = None
    path_index: int | None = None

    def __post_init__(self) -> None:
        _validate_point(self.reference, "command reference")
        if self.action not in ("move", "hover"):
            raise ValueError("action must be either 'move' or 'hover'")


@dataclass(frozen=True)
class ReconstructionAudit:
    """Diagnostics for checking reconstruction invariants."""

    message_timestamp: int
    current_time: int
    replayed_commands: int
    move_commands: int
    hover_commands: int
    legal_source_updates: int
    blocked_source_updates: int
    blocked_source_mass: float
    rejected_advance_mass: float


@dataclass(frozen=True)
class ReconstructionResult:
    """Spatial posterior and MAP planning anchor."""

    map_position: Point
    distribution: tuple[tuple[Point, float], ...]
    audit: ReconstructionAudit

    def as_dict(self) -> dict[Point, float]:
        return dict(self.distribution)

    @property
    def probability_mass(self) -> float:
        return fsum(probability for _, probability in self.distribution)


class ReachabilityConstrainedStateReconstructor:
    """Replay delayed feedback and issued commands on a legal motion graph."""

    def __init__(self, deviation_probability: float = 0.25) -> None:
        if not 0.0 <= deviation_probability <= 1.0:
            raise ValueError("deviation_probability must lie in [0, 1]")
        self.deviation_probability = float(deviation_probability)

    def reconstruct(
        self,
        message: StateMessage,
        ledger: Iterable[IssuedCommand],
        graph: MotionGraph,
        current_time: int,
    ) -> ReconstructionResult:
        """Reconstruct the spatial planning anchor at ``current_time``.

        Only commands strictly newer than the delivered message and no later
        than ``current_time`` are replayed. Commands with equal timestamps
        retain their ledger order.
        """

        if current_time < message.timestamp:
            raise ValueError("current_time cannot precede the message timestamp")
        if not graph.contains(message.position):
            raise ValueError("the delivered message position is not a graph vertex")

        indexed_commands = list(enumerate(ledger))
        commands = [
            (ledger_index, command)
            for ledger_index, command in indexed_commands
            if message.timestamp < command.timestamp <= current_time
        ]
        commands.sort(key=lambda item: (item[1].timestamp, item[0]))

        belief: dict[Point, float] = {message.position: 1.0}
        move_commands = 0
        hover_commands = 0
        legal_source_updates = 0
        blocked_source_updates = 0
        blocked_source_mass = 0.0
        rejected_advance_mass = 0.0

        for _, command in commands:
            if not graph.contains(command.reference):
                raise ValueError("an issued command reference is not a graph vertex")
            if command.action == "hover":
                hover_commands += 1
                continue

            move_commands += 1
            updated: dict[Point, float] = {}
            for position, probability in belief.items():
                if graph.has_edge(position, command.reference):
                    legal_source_updates += 1
                    _add_mass(
                        updated,
                        position,
                        probability * self.deviation_probability,
                    )
                    _add_mass(
                        updated,
                        command.reference,
                        probability * (1.0 - self.deviation_probability),
                    )
                else:
                    blocked_source_updates += 1
                    blocked_source_mass += probability
                    rejected_advance_mass += (
                        probability * (1.0 - self.deviation_probability)
                    )
                    _add_mass(updated, position, probability)

            _assert_probability_mass(updated)
            belief = updated

        distribution = tuple(
            sorted(belief.items(), key=lambda item: graph.rank(item[0]))
        )
        map_position = min(
            belief,
            key=lambda point: (-belief[point], graph.rank(point)),
        )
        audit = ReconstructionAudit(
            message_timestamp=message.timestamp,
            current_time=current_time,
            replayed_commands=len(commands),
            move_commands=move_commands,
            hover_commands=hover_commands,
            legal_source_updates=legal_source_updates,
            blocked_source_updates=blocked_source_updates,
            blocked_source_mass=blocked_source_mass,
            rejected_advance_mass=rejected_advance_mass,
        )
        return ReconstructionResult(map_position, distribution, audit)


def _validate_point(point: Point, label: str) -> None:
    if (
        not isinstance(point, tuple)
        or len(point) != 2
        or not all(isinstance(coordinate, int) for coordinate in point)
    ):
        raise TypeError(f"{label} must be a tuple of two integers")


def _add_mass(belief: dict[Point, float], point: Point, mass: float) -> None:
    if mass > 0.0:
        belief[point] = belief.get(point, 0.0) + mass


def _assert_probability_mass(belief: Mapping[Point, float]) -> None:
    if not belief or not isclose(
        fsum(belief.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("RCSR probability mass is not conserved")
