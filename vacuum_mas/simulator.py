"""
Simulator — The true grid world environment.

Implements the environment side of the agent–environment interface (AIMA §2.1).
The simulator holds the ground truth: which cells are free, which are blocked,
and where dirt is located. The agent never reads these directly — it can only
interact through sensors and actuators (§4.1, §4.2).

Environment properties (§1.3):
  - Partially observable (agent sees only current cell + forward cell)
  - Deterministic (same action in same state → same outcome)
  - Sequential (current actions affect future percepts)
  - Static (world does not change while agent deliberates)
  - Discrete (finite cells, 3 actions, 2 binary sensors)
  - Single-agent
"""

from typing import Dict, Set, Optional, Tuple, List
from .state import Cell, Direction
from .utils import get_neighbors

import random


class GridWorld:
    """
    True environment simulator.

    The grid uses a coordinate system where:
      - (0, 0) is the agent's start cell (always free)
      - x increases rightward (East)
      - y increases upward (North)

    Blocked cells include both perimeter walls and interior obstacles.
    The agent cannot distinguish between wall types (§4.5).
    """

    def __init__(
        self,
        width: int,
        height: int,
        obstacles: Set[Cell],
        dirt: Set[Cell],
        start: Cell = (0, 0),
    ):
        """
        Create a grid world.

        Args:
            width:     Grid width (x from 0 to width-1)
            height:    Grid height (y from 0 to height-1)
            obstacles: Set of blocked cells (interior obstacles)
            dirt:      Set of cells that initially contain dirt
            start:     Agent's starting cell (must be free)
        """
        self.width = width
        self.height = height
        self.start = start

        # Build the full set of blocked cells: obstacles + perimeter walls
        self.blocked: Set[Cell] = set(obstacles)
        for x in range(-1, width + 1):
            self.blocked.add((x, -1))
            self.blocked.add((x, height))
        for y in range(-1, height + 1):
            self.blocked.add((-1, y))
            self.blocked.add((width, y))

        # Dirt is mutable — removed when agent sucks (§4.1)
        self.dirt: Set[Cell] = set(dirt)

        # The true set of all free cells (for verification)
        self.free_cells: Set[Cell] = set()
        for x in range(width):
            for y in range(height):
                if (x, y) not in self.blocked:
                    self.free_cells.add((x, y))

        # Verify start is free
        assert start in self.free_cells, f"Start cell {start} is not free!"

        # Compute true reachable set R from start (for correctness verification)
        self.reachable: Set[Cell] = self._compute_reachable(start)

        # Statistics
        self.total_sucks = 0
        self.total_moves = 0
        self.total_turns = 0
        self.total_bumps = 0
        self.agent_positions: Dict[int, Cell] = {}
        self.agent_stats: Dict[int, Dict[str, int]] = {}

    def _compute_reachable(self, start: Cell) -> Set[Cell]:
        """
        Compute R — the true reachable free set from `start` via flood fill.

        This is the ground truth used to verify the agent's completeness
        at termination (§11.1: M = R when F = ∅).
        """
        from collections import deque

        visited = {start}
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            for neighbor in get_neighbors(cell):
                if neighbor in self.free_cells and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited

    # -------------------------------------------------------------------
    # Sensors (§4.2 — the minimal sensor pair)
    # -------------------------------------------------------------------

    def dirt_here(self, pos: Cell) -> bool:
        """
        Dirt sensor (§4.2): True iff current cell contains dirt.

        Sampled at each tick's start, before any action.
        """
        return pos in self.dirt

    def is_blocked(self, cell: Cell) -> bool:
        """
        Check if a cell is blocked (wall or obstacle).

        This is the ground truth — the agent accesses this only
        through bump or blocked_ahead sensors, never directly.
        """
        return cell in self.blocked

    def blocked_ahead(self, pos: Cell, heading: Direction) -> bool:
        """
        Lookahead sensor (§4.2 — Option B, recommended).

        Reports whether the forward cell is blocked WITHOUT the agent
        attempting to enter it. Enables the 4-state cell model (§3.5).
        """
        forward = (pos[0] + heading.dx, pos[1] + heading.dy)
        return self.is_blocked(forward)

    # -------------------------------------------------------------------
    # Actuators (§4.1 — the complete actuator set)
    # -------------------------------------------------------------------

    def register_agent(self, agent_id: int, start_pos: Cell) -> None:
        """Register an agent in the world for occupancy-aware movement."""
        assert start_pos in self.free_cells, f"Agent start {start_pos} not free"
        if any(p == start_pos for p in self.agent_positions.values()):
            raise ValueError(f"Cell {start_pos} already occupied by another agent")
        self.agent_positions[agent_id] = start_pos
        self.agent_stats.setdefault(
            agent_id, {"moves": 0, "turns": 0, "bumps": 0, "sucks": 0}
        )

    def try_move(
        self, pos: Cell, heading: Direction, agent_id: Optional[int] = None
    ) -> Tuple[Cell, bool]:
        """
        Attempt move_one_step (§4.1).

        Returns:
            (new_pos, bumped): new position and whether a bump occurred.

        Deterministic dynamics (§1.4): move succeeds iff forward cell is free.
        """
        forward = (pos[0] + heading.dx, pos[1] + heading.dy)

        if agent_id is not None:
            occupied = any(
                aid != agent_id and a_pos == forward
                for aid, a_pos in self.agent_positions.items()
            )
            if occupied:
                return pos, False

        if self.is_blocked(forward):
            self.total_bumps += 1
            if agent_id is not None:
                self.agent_stats.setdefault(
                    agent_id, {"moves": 0, "turns": 0, "bumps": 0, "sucks": 0}
                )
                self.agent_stats[agent_id]["bumps"] += 1
            return pos, True  # Bump — position unchanged
        else:
            self.total_moves += 1
            if agent_id is not None:
                self.agent_positions[agent_id] = forward
                self.agent_stats.setdefault(
                    agent_id, {"moves": 0, "turns": 0, "bumps": 0, "sucks": 0}
                )
                self.agent_stats[agent_id]["moves"] += 1
            return forward, False  # Success — moved to forward cell

    def suck(self, pos: Cell) -> bool:
        """
        Suck actuator (§4.1): remove dirt at current cell.

        Returns True if dirt was actually present and removed.
        Suck is idempotent — sucking a clean cell is a no-op (§4.1).
        """
        self.total_sucks += 1
        if pos in self.dirt:
            self.dirt.remove(pos)
            return True
        return False

    def suck_for_agent(self, agent_id: int, pos: Cell) -> bool:
        """Agent-aware suck action for per-agent stats tracking."""
        self.agent_stats.setdefault(
            agent_id, {"moves": 0, "turns": 0, "bumps": 0, "sucks": 0}
        )
        self.agent_stats[agent_id]["sucks"] += 1
        return self.suck(pos)

    def rotate_cw(self, heading: Direction) -> Direction:
        """
        change_direction actuator (§4.1): 90° clockwise rotation.

        N→E→S→W→N. This is the only turn primitive available.
        """
        self.total_turns += 1
        return heading.rotate_cw()

    def rotate_cw_for_agent(self, agent_id: int, heading: Direction) -> Direction:
        """Agent-aware clockwise turn for per-agent stats tracking."""
        self.agent_stats.setdefault(
            agent_id, {"moves": 0, "turns": 0, "bumps": 0, "sucks": 0}
        )
        self.agent_stats[agent_id]["turns"] += 1
        return self.rotate_cw(heading)

    def rotate_ccw(self, heading: Direction) -> Direction:
        """Counter-clockwise 90 degree rotation (TurnLeft actuator)."""
        self.total_turns += 1
        return heading.rotate_ccw()

    def rotate_ccw_for_agent(self, agent_id: int, heading: Direction) -> Direction:
        """Agent-aware counter-clockwise turn for per-agent stats tracking."""
        self.agent_stats.setdefault(
            agent_id, {"moves": 0, "turns": 0, "bumps": 0, "sucks": 0}
        )
        self.agent_stats[agent_id]["turns"] += 1
        return self.rotate_ccw(heading)

    def nearby_agents(self, pos: Cell, exclude_id: int) -> list:
        """Agent-presence sensor: return IDs of agents at pos or adjacent cells."""
        targets = {pos} | set(get_neighbors(pos))
        return [
            aid for aid, apos in self.agent_positions.items()
            if aid != exclude_id and apos in targets
        ]

    def nearby_agents_with_offset(
        self, pos: Cell, exclude_id: int,
    ) -> list:
        """Return [(agent_id, offset)] for nearby agents.
        offset = other_abs - my_abs (direction from me to other)."""
        targets = {pos} | set(get_neighbors(pos))
        result = []
        for aid, apos in self.agent_positions.items():
            if aid == exclude_id:
                continue
            if apos in targets:
                offset = (apos[0] - pos[0], apos[1] - pos[1])
                result.append((aid, offset))
        return result

    def is_cell_occupied_by_agent(self, cell: Cell, exclude_id: int) -> bool:
        """Check if any agent (other than exclude_id) occupies cell."""
        return any(
            aid != exclude_id and apos == cell
            for aid, apos in self.agent_positions.items()
        )

    # -------------------------------------------------------------------
    # Display helpers
    # -------------------------------------------------------------------

    def render_true_grid(self) -> str:
        """Render the true grid state (god's-eye view for debugging)."""
        lines = []
        for y in range(self.height - 1, -1, -1):
            row = []
            for x in range(self.width):
                cell = (x, y)
                if cell in self.blocked:
                    row.append("█")
                elif cell in self.dirt:
                    row.append("*")
                elif cell == self.start:
                    row.append("S")
                else:
                    row.append("·")
            lines.append(" ".join(row))
        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Return simulation statistics."""
        return {
            "total_moves": self.total_moves,
            "total_turns": self.total_turns,
            "total_sucks": self.total_sucks,
            "total_bumps": self.total_bumps,
            "total_actions": (
                self.total_moves + self.total_turns +
                self.total_sucks + self.total_bumps
            ),
            "true_reachable_cells": len(self.reachable),
            "true_free_cells": len(self.free_cells),
            "dirt_remaining": len(self.dirt),
            "per_agent": self.agent_stats,
        }


# ---------------------------------------------------------------------------
# Predefined grid layouts for testing (§6.6, testing strategy)
# ---------------------------------------------------------------------------

def create_single_room(width: int = 5, height: int = 5,
                       dirt_density: float = 0.3,
                       seed: int = 42) -> GridWorld:
    """
    Simple rectangular room with no internal obstacles.

    Good for verifying basic frontier expansion and cleaning.
    """
    rng = random.Random(seed)
    dirt = set()
    for x in range(width):
        for y in range(height):
            if (x, y) != (0, 0) and rng.random() < dirt_density:
                dirt.add((x, y))
    return GridWorld(width, height, obstacles=set(), dirt=dirt)


def create_two_rooms(seed: int = 42) -> GridWorld:
    """
    Two rooms connected by a single doorway.

    Layout (8×5):
        ████████
        █ Room █ Room █
        █  A   D  B   █
        █      █      █
        ████████████████

    D = doorway at (3, 2), wall at (3, 0), (3, 1), (3, 3), (3, 4)
    """
    width, height = 8, 5
    obstacles = set()
    # Internal wall with doorway
    for y in range(height):
        if y != 2:  # doorway at y=2
            obstacles.add((3, y))

    rng = random.Random(seed)
    dirt = set()
    for x in range(width):
        for y in range(height):
            if (x, y) not in obstacles and (x, y) != (0, 0):
                if rng.random() < 0.3:
                    dirt.add((x, y))

    return GridWorld(width, height, obstacles=obstacles, dirt=dirt)


def create_tree_of_rooms(seed: int = 42) -> GridWorld:
    """
    Three rooms in a tree topology (§9.2):

        Room A (left) ← doorway → Room B (center) ← doorway → Room C (right)

    Layout (12×5):
        ████████████
        █ A  █ B  █ C  █
        █    D    D    █
        █    █    █    █
        ████████████████
    """
    width, height = 12, 5
    obstacles = set()

    # Wall between A and B (x=3), doorway at y=2
    for y in range(height):
        if y != 2:
            obstacles.add((3, y))

    # Wall between B and C (x=7), doorway at y=2
    for y in range(height):
        if y != 2:
            obstacles.add((7, y))

    rng = random.Random(seed)
    dirt = set()
    for x in range(width):
        for y in range(height):
            if (x, y) not in obstacles and (x, y) != (0, 0):
                if rng.random() < 0.3:
                    dirt.add((x, y))

    return GridWorld(width, height, obstacles=obstacles, dirt=dirt)


def create_maze(seed: int = 42) -> GridWorld:
    """
    A small maze with narrow corridors — tests Warnsdorff heuristic.

    Layout (7×7):
        ███████
        █ · █ ·█
        █ ███ ·█
        █ · · ·█
        █ █ ████
        █ · · ·█
        ███████
    """
    width, height = 7, 7
    obstacles = {
        # Horizontal walls
        (1, 4), (2, 4), (3, 4),
        (3, 2), (4, 2), (5, 2),
        # Vertical walls
        (2, 5), (2, 4),
        (4, 3), (4, 4),
    }

    rng = random.Random(seed)
    dirt = set()
    for x in range(width):
        for y in range(height):
            if (x, y) not in obstacles and (x, y) != (0, 0):
                if rng.random() < 0.25:
                    dirt.add((x, y))

    return GridWorld(width, height, obstacles=obstacles, dirt=dirt)


def create_l_shaped_room(seed: int = 42) -> GridWorld:
    """
    L-shaped room — tests non-rectangular exploration.

    Layout (8×8):
        ████████
        █ · · · █
        █ · · · █
        █ · · · ████
        █ · · · · · █
        █ · · · · · █
        █ · · · · · █
        ██████████████
    """
    width, height = 8, 8
    obstacles = set()

    # Block upper-right quadrant
    for x in range(4, width):
        for y in range(5, height):
            obstacles.add((x, y))

    rng = random.Random(seed)
    dirt = set()
    for x in range(width):
        for y in range(height):
            if (x, y) not in obstacles and (x, y) != (0, 0):
                if rng.random() < 0.3:
                    dirt.add((x, y))

    return GridWorld(width, height, obstacles=obstacles, dirt=dirt)
