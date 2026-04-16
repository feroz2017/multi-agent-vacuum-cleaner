"""BFS paths and turn/move sequences on known-free cells M only."""

from collections import deque
from typing import Set, Dict, List, Optional, Tuple
from .state import Cell, Direction
from .utils import get_neighbors, heading_from


class NavigationError(Exception):
    pass


class Navigator:
    """Shortest paths on subgraph M; unknown cells are not used."""

    @staticmethod
    def bfs_path(start: Cell, goal: Cell, M: Set[Cell]) -> List[Cell]:
        if start == goal:
            return [start]

        queue: deque = deque()
        queue.append(start)
        came_from: Dict[Cell, Optional[Cell]] = {start: None}

        while queue:
            current = queue.popleft()

            for neighbor in get_neighbors(current):
                if neighbor in M and neighbor not in came_from:
                    came_from[neighbor] = current
                    if neighbor == goal:
                        path = []
                        node = goal
                        while node is not None:
                            path.append(node)
                            node = came_from[node]
                        path.reverse()
                        return path
                    queue.append(neighbor)

        raise NavigationError(
            f"Goal {goal} unreachable from {start} in known free space M "
            f"(|M| = {len(M)}). This should never happen if invariants hold."
        )

    @staticmethod
    def bfs_distances(start: Cell, M: Set[Cell]) -> Dict[Cell, int]:
        """BFS distance from start to each cell reachable in M."""
        dist: Dict[Cell, int] = {start: 0}
        queue: deque = deque([start])

        while queue:
            current = queue.popleft()
            d = dist[current]
            for neighbor in get_neighbors(current):
                if neighbor in M and neighbor not in dist:
                    dist[neighbor] = d + 1
                    queue.append(neighbor)

        return dist

    @staticmethod
    def path_to_actions(path: List[Cell], current_heading: Direction) -> List[str]:
        """Turn (cw) + move steps along path; returns \"TURN\" / \"MOVE\" strings."""
        actions = []
        heading = current_heading

        for i in range(len(path) - 1):
            desired = heading_from(path[i], path[i + 1])

            turns_needed = heading.turns_to(desired)
            for _ in range(turns_needed):
                actions.append("TURN")
                heading = heading.rotate_cw()

            actions.append("MOVE")

        return actions
