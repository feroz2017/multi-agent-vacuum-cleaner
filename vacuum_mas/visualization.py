"""ASCII grid of agent belief (optional frontier)."""

from __future__ import annotations

from typing import Optional, Set, Tuple

from .state import AgentState, Cell, CellState
from .simulator import GridWorld
from .frontier import compute_frontier


def render_belief_grid(
    world: GridWorld,
    state: AgentState,
    *,
    show_frontier: bool = True,
) -> str:
    """Text grid from M/O/U; agent as @."""
    F: Set[Cell] = set()
    if show_frontier:
        F = compute_frontier(
            state.M,
            state.O,
            state.U if state.use_lookahead else None,
        )

    lines = []
    for y in range(world.height - 1, -1, -1):
        row = []
        for x in range(world.width):
            c: Cell = (x, y)
            if (x, y) == (state.x, state.y):
                row.append("@")
                continue
            cs = state.cell_state(c)
            if cs == CellState.BLOCKED:
                ch = "#"
            elif cs == CellState.FREE_VISITED:
                ch = "."
            elif cs == CellState.FREE_UNVISITED:
                ch = "o"
            elif c in F:
                ch = "~"
            else:
                ch = "?"
            row.append(ch)
        lines.append(" ".join(f"{t:>2}" for t in row))
    return "\n".join(lines)


def render_true_grid(world: GridWorld, agent_pos: Optional[Cell] = None) -> str:
    """Compact true layout: █ wall/obstacle, * dirt, · free, S start, @ agent."""
    lines = []
    for y in range(world.height - 1, -1, -1):
        row = []
        for x in range(world.width):
            c = (x, y)
            if agent_pos is not None and c == agent_pos:
                row.append("@")
            elif c not in world.free_cells:
                row.append("█")
            elif c in world.dirt:
                row.append("*")
            elif c == world.start:
                row.append("S")
            else:
                row.append("·")
        lines.append(" ".join(row))
    return "\n".join(lines)


def print_step(agent, world: GridWorld, step_idx: int, action: str) -> None:
    print(f"\n=== Step {step_idx}: {action} ===")
    print(render_belief_grid(world, agent.state))
