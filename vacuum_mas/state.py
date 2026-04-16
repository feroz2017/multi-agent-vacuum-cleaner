"""Agent pose, heading, and M/O/U map sets."""

from enum import Enum
from dataclasses import dataclass, field
from typing import Set, Dict, Optional, Tuple

Cell = Tuple[int, int]


class Direction(Enum):
    """Heading; each member's value is (dx, dy) for one step."""
    NORTH = (0, +1)
    EAST  = (+1, 0)
    SOUTH = (0, -1)
    WEST  = (-1, 0)

    @property
    def dx(self) -> int:
        return self.value[0]

    @property
    def dy(self) -> int:
        return self.value[1]

    def rotate_cw(self) -> 'Direction':
        """Rotate 90° clockwise: N->E->S->W->N."""
        order = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        return order[(order.index(self) + 1) % 4]

    def rotate_ccw(self) -> 'Direction':
        """Rotate 90° counter-clockwise: N->W->S->E->N."""
        order = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        return order[(order.index(self) - 1) % 4]

    def turns_to(self, target: 'Direction') -> int:
        """Clockwise quarter-turns needed to face target."""
        order = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        return (order.index(target) - order.index(self)) % 4


class CellState(Enum):
    """How the agent labels a cell: unknown, blocked, visited, or seen-free-not-entered."""
    UNKNOWN        = "?"
    BLOCKED        = "#"
    FREE_VISITED   = "."
    FREE_UNVISITED = "o"


@dataclass
class AgentState:
    """Pose plus M (visited free), O (blocked), U (lookahead free-not-visited)."""

    agent_id: int = 0
    x: int = 0
    y: int = 0
    heading: Direction = Direction.NORTH

    M: Set[Cell] = field(default_factory=lambda: {(0, 0)})
    O: Set[Cell] = field(default_factory=set)
    U: Set[Cell] = field(default_factory=set)

    parent: Dict[Cell, Optional[Cell]] = field(
        default_factory=lambda: {(0, 0): None}
    )

    use_lookahead: bool = True

    @property
    def pos(self) -> Cell:
        return (self.x, self.y)

    @pos.setter
    def pos(self, value: Cell) -> None:
        self.x, self.y = value

    def forward_cell(self) -> Cell:
        return (self.x + self.heading.dx, self.y + self.heading.dy)

    def cell_state(self, cell: Cell) -> CellState:
        if cell in self.M:
            return CellState.FREE_VISITED
        if cell in self.O:
            return CellState.BLOCKED
        if cell in self.U:
            return CellState.FREE_UNVISITED
        return CellState.UNKNOWN

    def classify_as_free(self, cell: Cell) -> None:
        self.M.add(cell)
        self.U.discard(cell)

    def classify_as_blocked(self, cell: Cell) -> None:
        self.O.add(cell)

    def classify_as_free_unvisited(self, cell: Cell) -> None:
        if cell not in self.M and cell not in self.O:
            self.U.add(cell)

    def check_invariants(self) -> None:
        assert self.pos in self.M, \
            f"Invariant violation: agent at {self.pos} but pos not in M"

        assert len(self.M & self.O) == 0, \
            f"Invariant violation: M ∩ O ≠ ∅: {self.M & self.O}"

        assert len(self.U & self.M) == 0, \
            f"Invariant violation: U ∩ M ≠ ∅"
        assert len(self.U & self.O) == 0, \
            f"Invariant violation: U ∩ O ≠ ∅"
