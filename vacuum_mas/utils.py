"""Grid helpers: 4-neighbors, Manhattan distance, step heading."""

from typing import List, Tuple
from .state import Cell, Direction

NEIGHBOR_DELTAS: List[Tuple[int, int]] = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def get_neighbors(cell: Cell) -> List[Cell]:
    """Orthogonal neighbors of cell (N/E/S/W only, no diagonals)."""
    x, y = cell
    return [(x + dx, y + dy) for dx, dy in NEIGHBOR_DELTAS]


def is_adjacent(c1: Cell, c2: Cell) -> bool:
    return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) == 1


def manhattan_distance(c1: Cell, c2: Cell) -> int:
    return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])


def heading_from(from_cell: Cell, to_cell: Cell) -> Direction:
    """Heading for one orthogonal step from_cell -> to_cell."""
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
    return (cell[0] + direction.dx, cell[1] + direction.dy)
