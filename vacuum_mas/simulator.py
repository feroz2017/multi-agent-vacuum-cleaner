"""GridWorld: walls, dirt, movement; sensors/actuators used by agents."""

from typing import Dict, Set, Optional, Tuple, List
from .state import Cell, Direction
from .utils import get_neighbors

import random


class GridWorld:
    """Discrete grid: x right, y up; perimeter is blocked."""

    def __init__(
        self,
        width: int,
        height: int,
        obstacles: Set[Cell],
        dirt: Set[Cell],
        start: Cell = (0, 0),
    ):
        self.width = width
        self.height = height
        self.start = start

        self.blocked: Set[Cell] = set(obstacles)
        for x in range(-1, width + 1):
            self.blocked.add((x, -1))
            self.blocked.add((x, height))
        for y in range(-1, height + 1):
            self.blocked.add((-1, y))
            self.blocked.add((width, y))

        self.dirt: Set[Cell] = set(dirt)

        self.free_cells: Set[Cell] = set()
        for x in range(width):
            for y in range(height):
                if (x, y) not in self.blocked:
                    self.free_cells.add((x, y))

        assert start in self.free_cells, f"Start cell {start} is not free!"

        self.reachable: Set[Cell] = self._compute_reachable(start)

        self.total_sucks = 0
        self.total_moves = 0
        self.total_turns = 0
        self.total_bumps = 0
        self.agent_positions: Dict[int, Cell] = {}
        self.agent_stats: Dict[int, Dict[str, int]] = {}

    def _compute_reachable(self, start: Cell) -> Set[Cell]:
        """Flood fill free cells reachable from start."""
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

    def dirt_here(self, pos: Cell) -> bool:
        return pos in self.dirt

    def is_blocked(self, cell: Cell) -> bool:
        return cell in self.blocked

    def blocked_ahead(self, pos: Cell, heading: Direction) -> bool:
        forward = (pos[0] + heading.dx, pos[1] + heading.dy)
        return self.is_blocked(forward)

    def register_agent(self, agent_id: int, start_pos: Cell) -> None:
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
        """Step forward if free; return (new_pos, bumped)."""
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
            return pos, True
        else:
            self.total_moves += 1
            if agent_id is not None:
                self.agent_positions[agent_id] = forward
                self.agent_stats.setdefault(
                    agent_id, {"moves": 0, "turns": 0, "bumps": 0, "sucks": 0}
                )
                self.agent_stats[agent_id]["moves"] += 1
            return forward, False

    def suck(self, pos: Cell) -> bool:
        """Remove dirt at pos; returns whether dirt was there."""
        self.total_sucks += 1
        if pos in self.dirt:
            self.dirt.remove(pos)
            return True
        return False

    def suck_for_agent(self, agent_id: int, pos: Cell) -> bool:
        self.agent_stats.setdefault(
            agent_id, {"moves": 0, "turns": 0, "bumps": 0, "sucks": 0}
        )
        self.agent_stats[agent_id]["sucks"] += 1
        return self.suck(pos)

    def rotate_cw(self, heading: Direction) -> Direction:
        self.total_turns += 1
        return heading.rotate_cw()

    def rotate_cw_for_agent(self, agent_id: int, heading: Direction) -> Direction:
        self.agent_stats.setdefault(
            agent_id, {"moves": 0, "turns": 0, "bumps": 0, "sucks": 0}
        )
        self.agent_stats[agent_id]["turns"] += 1
        return self.rotate_cw(heading)

    def rotate_ccw(self, heading: Direction) -> Direction:
        self.total_turns += 1
        return heading.rotate_ccw()

    def rotate_ccw_for_agent(self, agent_id: int, heading: Direction) -> Direction:
        self.agent_stats.setdefault(
            agent_id, {"moves": 0, "turns": 0, "bumps": 0, "sucks": 0}
        )
        self.agent_stats[agent_id]["turns"] += 1
        return self.rotate_ccw(heading)

    def nearby_agents(self, pos: Cell, exclude_id: int) -> list:
        targets = {pos} | set(get_neighbors(pos))
        return [
            aid for aid, apos in self.agent_positions.items()
            if aid != exclude_id and apos in targets
        ]

    def nearby_agents_with_offset(
        self, pos: Cell, exclude_id: int,
    ) -> list:
        """Peers at pos or neighbors; offset is their pos minus pos."""
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
        return any(
            aid != exclude_id and apos == cell
            for aid, apos in self.agent_positions.items()
        )

    def render_true_grid(self) -> str:
        """ASCII map for debugging."""
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


def create_single_room(width: int = 5, height: int = 5,
                       dirt_density: float = 0.3,
                       seed: int = 42) -> GridWorld:
    rng = random.Random(seed)
    dirt = set()
    for x in range(width):
        for y in range(height):
            if (x, y) != (0, 0) and rng.random() < dirt_density:
                dirt.add((x, y))
    return GridWorld(width, height, obstacles=set(), dirt=dirt)


def create_two_rooms(seed: int = 42) -> GridWorld:
    width, height = 8, 5
    obstacles = set()
    for y in range(height):
        if y != 2:
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
    width, height = 12, 5
    obstacles = set()

    for y in range(height):
        if y != 2:
            obstacles.add((3, y))

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
    width, height = 7, 7
    obstacles = {
        (1, 4), (2, 4), (3, 4),
        (3, 2), (4, 2), (5, 2),
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
    width, height = 8, 8
    obstacles = set()

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
