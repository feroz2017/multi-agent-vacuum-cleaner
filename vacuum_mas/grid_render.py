"""Pygame helpers: true map, belief map, legend, stats."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import pygame

from .state import Direction, Cell, CellState

if TYPE_CHECKING:
    from .state import AgentState
    from .simulator import GridWorld


COLORS = {
    "bg":             (30, 33, 40),
    "panel_bg":       (38, 42, 52),
    "wall":           (55, 60, 72),
    "floor":          (210, 215, 225),
    "dirt":           (190, 140, 50),
    "dirt_outline":   (150, 105, 30),
    "start":          (170, 195, 240),
    "agent":          (40, 160, 220),
    "agent_outline":  (20, 90, 140),
    "grid_line":      (170, 175, 185),
    "visited":        (190, 220, 190),
    "blocked":        (55, 60, 72),
    "unvisited":      (180, 200, 240),
    "frontier":       (100, 200, 100),
    "frontier_ring":  (60, 170, 60),
    "unknown":        (85, 88, 100),
    "path":           (255, 180, 80),
    "path_dot":       (230, 140, 40),
    "target":         (255, 80, 80),
    "text":           (220, 225, 235),
    "text_dim":       (140, 145, 155),
    "text_heading":   (255, 220, 100),
    "divider":        (60, 65, 78),
    "legend_bg":      (42, 46, 56),
}


def _cell_rect(
    gx: int, gy: int, height: int, cell: int, offset_x: int = 0, offset_y: int = 0
) -> pygame.Rect:
    """Cell rect; grid y-up to screen y-down."""
    row = height - 1 - gy
    return pygame.Rect(offset_x + gx * cell, offset_y + row * cell, cell, cell)


def _draw_agent(
    surface: pygame.Surface,
    rect: pygame.Rect,
    heading: Direction,
    color: Tuple[int, int, int] = COLORS["agent"],
    outline: Tuple[int, int, int] = COLORS["agent_outline"],
) -> None:
    cx, cy = rect.center
    r = max(5, rect.width // 3)

    if heading == Direction.NORTH:
        pts = [(cx, cy - r), (cx - r, cy + r * 2 // 3), (cx + r, cy + r * 2 // 3)]
    elif heading == Direction.SOUTH:
        pts = [(cx, cy + r), (cx - r, cy - r * 2 // 3), (cx + r, cy - r * 2 // 3)]
    elif heading == Direction.EAST:
        pts = [(cx + r, cy), (cx - r * 2 // 3, cy - r), (cx - r * 2 // 3, cy + r)]
    else:  # WEST
        pts = [(cx - r, cy), (cx + r * 2 // 3, cy - r), (cx + r * 2 // 3, cy + r)]

    pygame.draw.polygon(surface, color, pts)
    pygame.draw.polygon(surface, outline, pts, 2)


def draw_true_world(
    surface: pygame.Surface,
    world: "GridWorld",
    agent_pos: Cell,
    heading: Direction,
    cell_size: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> None:
    h, w = world.height, world.width
    cs = cell_size

    for gx in range(w):
        for gy in range(h):
            rect = _cell_rect(gx, gy, h, cs, offset_x, offset_y)
            c = (gx, gy)
            if c not in world.free_cells:
                pygame.draw.rect(surface, COLORS["wall"], rect)
            else:
                col = COLORS["start"] if c == world.start else COLORS["floor"]
                pygame.draw.rect(surface, col, rect)
                if c in world.dirt:
                    inset = cs // 4
                    dirt_rect = rect.inflate(-inset * 2, -inset * 2)
                    pygame.draw.rect(surface, COLORS["dirt"], dirt_rect, border_radius=cs // 6)
                    pygame.draw.rect(surface, COLORS["dirt_outline"], dirt_rect, 1, border_radius=cs // 6)
            pygame.draw.rect(surface, COLORS["grid_line"], rect, 1)

    agent_rect = _cell_rect(agent_pos[0], agent_pos[1], h, cs, offset_x, offset_y)
    _draw_agent(surface, agent_rect, heading)


def draw_belief_map(
    surface: pygame.Surface,
    world: "GridWorld",
    state: "AgentState",
    frontier: Set[Cell],
    path_cells: List[Cell],
    target_frontier: Optional[Cell],
    cell_size: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> None:
    h, w = world.height, world.width
    cs = cell_size
    path_set = set(path_cells)

    for gx in range(w):
        for gy in range(h):
            rect = _cell_rect(gx, gy, h, cs, offset_x, offset_y)
            c = (gx, gy)

            cs_state = state.cell_state(c)
            if cs_state == CellState.FREE_VISITED:
                fill = COLORS["visited"]
            elif cs_state == CellState.BLOCKED:
                fill = COLORS["blocked"]
            elif cs_state == CellState.FREE_UNVISITED:
                fill = COLORS["unvisited"]
            elif c in frontier:
                fill = COLORS["frontier"]
            else:
                fill = COLORS["unknown"]

            pygame.draw.rect(surface, fill, rect)

            if c in frontier:
                pygame.draw.rect(surface, COLORS["frontier_ring"], rect, 2)

            if c in path_set and c != state.pos:
                dot_r = max(3, cs // 8)
                pygame.draw.circle(surface, COLORS["path"], rect.center, dot_r)

            if target_frontier and c == target_frontier:
                pygame.draw.rect(surface, COLORS["target"], rect, 3)

            if c in state.M and c in world.dirt:
                dot_r = max(2, cs // 10)
                pygame.draw.circle(surface, COLORS["dirt"], rect.center, dot_r)

            pygame.draw.rect(surface, COLORS["grid_line"], rect, 1)

    if len(path_cells) >= 2:
        pts = []
        for pc in path_cells:
            r = _cell_rect(pc[0], pc[1], h, cs, offset_x, offset_y)
            pts.append(r.center)
        pygame.draw.lines(surface, COLORS["path"], False, pts, 2)

    agent_rect = _cell_rect(state.x, state.y, h, cs, offset_x, offset_y)
    _draw_agent(surface, agent_rect, state.heading)


def draw_true_world_multi(
    surface: pygame.Surface,
    world: "GridWorld",
    agents: List[Tuple[Cell, Direction, Tuple[int, int, int]]],
    cell_size: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> None:
    h, w = world.height, world.width
    cs = cell_size
    for gx in range(w):
        for gy in range(h):
            rect = _cell_rect(gx, gy, h, cs, offset_x, offset_y)
            c = (gx, gy)
            if c not in world.free_cells:
                pygame.draw.rect(surface, COLORS["wall"], rect)
            else:
                col = COLORS["start"] if c == world.start else COLORS["floor"]
                pygame.draw.rect(surface, col, rect)
                if c in world.dirt:
                    inset = cs // 4
                    dirt_rect = rect.inflate(-inset * 2, -inset * 2)
                    pygame.draw.rect(surface, COLORS["dirt"], dirt_rect, border_radius=cs // 6)
                    pygame.draw.rect(surface, COLORS["dirt_outline"], dirt_rect, 1, border_radius=cs // 6)
            pygame.draw.rect(surface, COLORS["grid_line"], rect, 1)
    for pos, heading, color in agents:
        rect = _cell_rect(pos[0], pos[1], h, cs, offset_x, offset_y)
        _draw_agent(surface, rect, heading, color=color)


def draw_belief_map_multi(
    surface: pygame.Surface,
    world: "GridWorld",
    M: Set[Cell],
    O: Set[Cell],
    U: Set[Cell],
    frontier_assignments: Dict[int, Set[Cell]],
    agents: List[Tuple[int, Cell, Direction, Tuple[int, int, int]]],
    path_cells: Dict[int, List[Cell]],
    target_frontiers: Dict[int, Optional[Cell]],
    cell_size: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> None:
    h, w = world.height, world.width
    cs = cell_size
    all_frontiers: Set[Cell] = set()
    for fset in frontier_assignments.values():
        all_frontiers |= set(fset)

    frontier_owner: Dict[Cell, int] = {}
    for aid, fset in frontier_assignments.items():
        for cell in fset:
            frontier_owner[cell] = aid
    agent_colors = {aid: color for aid, _, _, color in agents}

    for gx in range(w):
        for gy in range(h):
            rect = _cell_rect(gx, gy, h, cs, offset_x, offset_y)
            c = (gx, gy)
            if c in M:
                fill = COLORS["visited"]
            elif c in O:
                fill = COLORS["blocked"]
            elif c in U:
                fill = COLORS["unvisited"]
            elif c in all_frontiers:
                fill = COLORS["frontier"]
            else:
                fill = COLORS["unknown"]
            pygame.draw.rect(surface, fill, rect)
            owner = frontier_owner.get(c)
            if owner is not None:
                pygame.draw.rect(surface, agent_colors.get(owner, COLORS["frontier_ring"]), rect, 2)
            pygame.draw.rect(surface, COLORS["grid_line"], rect, 1)

    for aid, path in path_cells.items():
        if len(path) < 2:
            continue
        col = agent_colors.get(aid, COLORS["path"])
        pts = []
        for p in path:
            r = _cell_rect(p[0], p[1], h, cs, offset_x, offset_y)
            pts.append(r.center)
        pygame.draw.lines(surface, col, False, pts, 2)

    for aid, target in target_frontiers.items():
        if not target:
            continue
        col = agent_colors.get(aid, COLORS["target"])
        tr = _cell_rect(target[0], target[1], h, cs, offset_x, offset_y)
        pygame.draw.rect(surface, col, tr, 3)

    for _, pos, heading, color in agents:
        rect = _cell_rect(pos[0], pos[1], h, cs, offset_x, offset_y)
        _draw_agent(surface, rect, heading, color=color)


def draw_panel_label(
    surface: pygame.Surface,
    text: str,
    x: int,
    y: int,
    font: pygame.font.Font,
    color: Tuple[int, int, int] = COLORS["text"],
) -> None:
    rendered = font.render(text, True, color)
    rect = rendered.get_rect(midtop=(x, y))
    surface.blit(rendered, rect)


_LEGEND_ITEMS = [
    ("Visited (M)", "visited"),
    ("Blocked (O)", "blocked"),
    ("Free-Unvisited (U)", "unvisited"),
    ("Frontier (F)", "frontier"),
    ("Unknown", "unknown"),
    ("BFS Path", "path"),
    ("Target", "target"),
    ("Dirt", "dirt"),
    ("Agent", "agent"),
]


def draw_legend(
    surface: pygame.Surface,
    y: int,
    width: int,
    font: pygame.font.Font,
) -> None:
    pygame.draw.rect(surface, COLORS["legend_bg"], (0, y, width, 30))
    pygame.draw.line(surface, COLORS["divider"], (0, y), (width, y))

    x = 12
    for label, color_key in _LEGEND_ITEMS:
        color = COLORS[color_key]
        pygame.draw.rect(surface, color, (x, y + 8, 14, 14), border_radius=3)
        pygame.draw.rect(surface, COLORS["text_dim"], (x, y + 8, 14, 14), 1, border_radius=3)
        x += 18
        rendered = font.render(label, True, COLORS["text_dim"])
        surface.blit(rendered, (x, y + 8))
        x += rendered.get_width() + 16


def draw_stats_panel(
    surface: pygame.Surface,
    x: int,
    y: int,
    width: int,
    height: int,
    stats: Dict[str, str],
    font: pygame.font.Font,
    title_font: pygame.font.Font,
) -> None:
    pygame.draw.rect(surface, COLORS["panel_bg"], (x, y, width, height))
    pygame.draw.rect(surface, COLORS["divider"], (x, y, width, height), 1)

    ty = y + 8
    title = title_font.render("Agent Status", True, COLORS["text_heading"])
    surface.blit(title, (x + 10, ty))
    ty += 28

    for key, value in stats.items():
        key_surf = font.render(f"{key}:", True, COLORS["text_dim"])
        val_surf = font.render(str(value), True, COLORS["text"])
        surface.blit(key_surf, (x + 10, ty))
        surface.blit(val_surf, (x + 10 + key_surf.get_width() + 6, ty))
        ty += 20


def init_pygame() -> None:
    pygame.init()
    pygame.display.set_caption("Vacuum Cleaning Agent")
