"""
Navigation — Pathfinding in known free space.

Implements BFS on the known subgraph (Document §07).
The agent navigates ONLY through cells in M (visited free cells).
This constraint is essential for the completeness proof (§6.6):
navigating through unknown cells could corrupt position estimates.

AIMA §3.4.1 — Breadth-First Search:
  - Optimal for unweighted graphs (finds shortest path)
  - Complete (always finds a path if one exists)
  - O(|M|) time per query
"""

from collections import deque
from typing import Set, Dict, List, Optional, Tuple
from .state import Cell, Direction
from .utils import get_neighbors, heading_from


class NavigationError(Exception):
    """Raised when navigation fails (goal unreachable in M)."""
    pass


class Navigator:
    """
    Navigate through known free space using BFS (§7.2).

    All paths are constrained to cells in M — the agent never
    plans a route through unknown or blocked cells.
    """

    @staticmethod
    def bfs_path(start: Cell, goal: Cell, M: Set[Cell]) -> List[Cell]:
        """
        Find the shortest path from `start` to `goal` using only M cells.

        Args:
            start: Source cell (must be in M)
            goal:  Target cell (must be in M)
            M:     Set of visited free cells (navigable space)

        Returns:
            List of cells [start, ..., goal] forming the shortest path.

        Raises:
            NavigationError if goal is unreachable from start in M.

        Complexity: O(|M|) — standard BFS on an unweighted graph.

        References: §7.2 (BFS on known subgraph), AIMA §3.4.1
        """
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
                        # Reconstruct path
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
        """
        Compute BFS distances from `start` to all reachable cells in M.

        Used by frontier selection heuristics (§8.2 — nearest frontier)
        to efficiently find distances to all M-neighbors of frontier cells.

        Returns:
            Dict mapping each reachable cell in M to its BFS distance from start.

        Complexity: O(|M|) — single-source BFS.
        """
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
        """
        Convert a cell path to a sequence of primitive actions (§7.3).

        Each step (u → v) requires:
          1. Turn to face v (0-3 change_direction calls)
          2. move_one_step

        Returns:
            List of action strings: "TURN", "MOVE"

        Turn cost analysis (§7.3): For path of k cells, at most 3(k-1)
        turns worst case. Boustrophedon paths have mostly 0-turn moves.
        """
        actions = []
        heading = current_heading

        for i in range(len(path) - 1):
            desired = heading_from(path[i], path[i + 1])

            # Turn toward desired heading (clockwise only)
            turns_needed = heading.turns_to(desired)
            for _ in range(turns_needed):
                actions.append("TURN")
                heading = heading.rotate_cw()

            actions.append("MOVE")

        return actions
