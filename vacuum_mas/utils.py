"""
Utils — Coordinate math and helper functions.

Provides 4-connectivity neighbor queries, distance metrics, and
heading computation used throughout the agent implementation.

All functions operate on Cell = Tuple[int, int] for zero-overhead hashing.
"""

from typing import List, Tuple
from .state import Cell, Direction


# ---------------------------------------------------------------------------
# 4-connectivity neighbors (§3.1 — grid adjacency contract)
# ---------------------------------------------------------------------------

# Pre-computed deltas: N, E, S, W
NEIGHBOR_DELTAS: List[Tuple[int, int]] = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def get_neighbors(cell: Cell) -> List[Cell]:
    """
    Return the four 4-adjacent neighbors of `cell`.

    Grid adjacency contract (§1.4): legal moves are exactly {N, E, S, W}.
    Diagonal moves are not supported.
    """
    x, y = cell
    return [(x + dx, y + dy) for dx, dy in NEIGHBOR_DELTAS]


def is_adjacent(c1: Cell, c2: Cell) -> bool:
    """Check if two cells are 4-adjacent (Manhattan distance == 1)."""
    return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) == 1


def manhattan_distance(c1: Cell, c2: Cell) -> int:
    """Manhattan distance between two cells."""
    return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])


# ---------------------------------------------------------------------------
# Heading computation (§7.3 — converting cell paths to primitive actions)
# ---------------------------------------------------------------------------

def heading_from(from_cell: Cell, to_cell: Cell) -> Direction:
    """
    Determine the heading needed to move from `from_cell` to `to_cell`.

    Precondition: cells must be 4-adjacent.
    Used by NavigateTo (§7.6) and ExecutePath (§7.3).
    """
    dx = to_cell[0] - from_cell[0]
    dy = to_cell[1] - from_cell[1]

    if dy == +1:
        return Direction.NORTH
    elif dx == +1:
        return Direction.EAST
    elif dy == -1:
        return Direction.SOUTH
    elif dx == -1:
        return Direction.WEST
    else:
        raise ValueError(
            f"Cells {from_cell} and {to_cell} are not 4-adjacent"
        )


def step_in_direction(cell: Cell, direction: Direction) -> Cell:
    """Return the cell one step from `cell` in `direction`."""
    return (cell[0] + direction.dx, cell[1] + direction.dy)
