"""
State — Data structures for the Vacuum Cleaning Agent.

Implements the internal state model from Document §05 of the Elaborated Solution.
Aligned with AIMA 4th Edition §2.4.4 (Model-Based Reflex Agent).

State vector: (x, y, theta, M, O, U, parent)
  - (x, y):   Dead-reckoned position (§3.3)
  - theta:    Current heading {N, E, S, W}
  - M:        Visited free cells (monotonically growing)
  - O:        Known blocked cells (monotonically growing)
  - U:        Free-unvisited cells (lookahead sensor only, §3.5)
  - parent:   Discovery tree for DFS narrative (§5.3)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Set, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Type alias: a cell is simply an (x, y) integer tuple.
# Using tuples keeps things hashable and set-compatible without overhead.
# ---------------------------------------------------------------------------
Cell = Tuple[int, int]


class Direction(Enum):
    """
    Agent heading — four cardinal directions.

    Each value is the (dx, dy) delta for one step in that direction.
    Clockwise rotation order: N → E → S → W → N  (§4.1)
    """
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
        """Number of clockwise 90° turns needed to face `target`."""
        order = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        return (order.index(target) - order.index(self)) % 4


class CellState(Enum):
    """
    Cell classification from the agent's perspective (§3.5 — four-state model).

    UNKNOWN:        No evidence acquired yet (absent from all sets).
    BLOCKED:        Bump received or blocked_ahead sensed.
    FREE_VISITED:   Agent has stepped onto this cell.
    FREE_UNVISITED: blocked_ahead=false sensed but not entered (lookahead only).
    """
    UNKNOWN        = "?"
    BLOCKED        = "#"
    FREE_VISITED   = "."
    FREE_UNVISITED = "o"


@dataclass
class AgentState:
    """
    Complete internal state of the model-based reflex agent (§5.1).

    This is the 'model' in AIMA's model-based reflex architecture (§2.4.4).
    Knowledge grows monotonically — cells are never removed from M or O (§3.6).
    """

    # --- Pose (dead-reckoned, §3.3) ---
    agent_id: int = 0
    x: int = 0
    y: int = 0
    heading: Direction = Direction.NORTH

    # --- Map sets (§3.4 — sparse hash storage) ---
    M: Set[Cell] = field(default_factory=lambda: {(0, 0)})       # visited free
    O: Set[Cell] = field(default_factory=set)                     # known blocked
    U: Set[Cell] = field(default_factory=set)                     # free-unvisited (lookahead)

    # --- Discovery tree (§5.3 — optional, for DFS narrative) ---
    parent: Dict[Cell, Optional[Cell]] = field(
        default_factory=lambda: {(0, 0): None}
    )

    # --- Configuration ---
    use_lookahead: bool = True  # Which sensor variant (§4.2)

    @property
    def pos(self) -> Cell:
        """Current position as a cell tuple."""
        return (self.x, self.y)

    @pos.setter
    def pos(self, value: Cell) -> None:
        self.x, self.y = value

    def forward_cell(self) -> Cell:
        """Cell the agent is currently facing (one step ahead)."""
        return (self.x + self.heading.dx, self.y + self.heading.dy)

    def cell_state(self, cell: Cell) -> CellState:
        """Classify a cell from the agent's perspective (§3.5)."""
        if cell in self.M:
            return CellState.FREE_VISITED
        if cell in self.O:
            return CellState.BLOCKED
        if cell in self.U:
            return CellState.FREE_UNVISITED
        return CellState.UNKNOWN

    def classify_as_free(self, cell: Cell) -> None:
        """
        Mark cell as visited-free (entered it). Monotonic — §3.6.

        Invariant: standing implies free (consistency rule 1).
        """
        self.M.add(cell)
        self.U.discard(cell)  # promote from unvisited if present

    def classify_as_blocked(self, cell: Cell) -> None:
        """
        Mark cell as blocked (bump received). Monotonic — §3.6.

        Invariant: bump implies blocked (consistency rule 2).
        """
        self.O.add(cell)

    def classify_as_free_unvisited(self, cell: Cell) -> None:
        """
        Mark cell as free-unvisited (lookahead sensor, §3.5).

        Only applicable with the 4-state model (Option B).
        """
        if cell not in self.M and cell not in self.O:
            self.U.add(cell)

    def check_invariants(self) -> None:
        """
        Debug assertions — verify monotonic model invariants (§3.6, §10.5).

        Call after every tick during development.
        """
        # 1. Standing implies free
        assert self.pos in self.M, \
            f"Invariant violation: agent at {self.pos} but pos not in M"

        # 2. M and O are disjoint
        assert len(self.M & self.O) == 0, \
            f"Invariant violation: M ∩ O ≠ ∅: {self.M & self.O}"

        # 3. U is disjoint from M and O
        assert len(self.U & self.M) == 0, \
            f"Invariant violation: U ∩ M ≠ ∅"
        assert len(self.U & self.O) == 0, \
            f"Invariant violation: U ∩ O ≠ ∅"
