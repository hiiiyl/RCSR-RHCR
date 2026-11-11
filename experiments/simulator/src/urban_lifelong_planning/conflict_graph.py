"""Future-path and AoI-tube conflicts represented as an UAV graph."""
from __future__ import annotations

from dataclasses import dataclass

from .aoi_occupancy_tube import tube_conflict_count
from .rhcr_adapted import _diagonals_cross

Point = tuple[int, int]


@dataclass(frozen=True)
class ConflictGraph:
    drone_count: int
    edges: frozenset[tuple[int, int]]

    def components(self) -> list[tuple[int, ...]]:
        adjacency = {index: set() for index in range(self.drone_count)}
        for left, right in self.edges:
            adjacency[left].add(right)
            adjacency[right].add(left)
        seen: set[int] = set()
        components: list[tuple[int, ...]] = []
        for root in range(self.drone_count):
            if root in seen or not adjacency[root]:
                continue
            stack, component = [root], []
            seen.add(root)
            while stack:
                node = stack.pop()
                component.append(node)
                for neighbor in adjacency[node] - seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
            components.append(tuple(sorted(component)))
        return sorted(components)


def build_conflict_graph(paths: list[list[Point]], tubes: list[list[set[Point]]]) -> ConflictGraph:
    edges: set[tuple[int, int]] = set()
    for left in range(len(paths)):
        for right in range(left + 1, len(paths)):
            if _paths_conflict(paths[left], paths[right]) or any(a & b for a, b in zip(tubes[left], tubes[right])):
                edges.add((left, right))
    return ConflictGraph(len(paths), frozenset(edges))


def _paths_conflict(left: list[Point], right: list[Point]) -> bool:
    for time in range(min(len(left), len(right))):
        if left[time] == right[time]:
            return True
        if time and (left[time - 1] == right[time] and right[time - 1] == left[time] or _diagonals_cross(left[time - 1], left[time], right[time - 1], right[time])):
            return True
    return False
