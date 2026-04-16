"""Procedural caves, floor plans, warehouse layouts."""

from __future__ import annotations

import random
from typing import List, Set, Tuple

from .simulator import GridWorld
from .state import Cell


def _scatter_dirt(
    free_cells: Set[Cell], start: Cell, dirt_density: float, rng: random.Random
) -> Set[Cell]:
    dirt: Set[Cell] = set()
    for cell in free_cells:
        if cell == start:
            continue
        if rng.random() < dirt_density:
            dirt.add(cell)
    return dirt


def create_cave(
    width: int = 30,
    height: int = 20,
    wall_prob: float = 0.45,
    iterations: int = 5,
    dirt_density: float = 0.2,
    seed: int = 42,
) -> GridWorld:
    rng = random.Random(seed)
    start = (0, 0)
    walls: Set[Cell] = set()
    for x in range(width):
        for y in range(height):
            if (x, y) == start:
                continue
            if rng.random() < wall_prob:
                walls.add((x, y))

    def wall_neighbors(cell: Cell) -> int:
        x, y = cell
        count = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 0 or nx >= width or ny >= height or (nx, ny) in walls:
                    count += 1
        return count

    for _ in range(iterations):
        next_walls: Set[Cell] = set()
        for x in range(width):
            for y in range(height):
                c = (x, y)
                if c == start:
                    continue
                if wall_neighbors(c) >= 5:
                    next_walls.add(c)
        walls = next_walls

    free = {(x, y) for x in range(width) for y in range(height) if (x, y) not in walls}
    if start not in free:
        free.add(start)
        walls.discard(start)

    # keep largest connected free region
    seen: Set[Cell] = set()
    components: List[Set[Cell]] = []
    for cell in sorted(free):
        if cell in seen:
            continue
        comp: Set[Cell] = set()
        stack = [cell]
        seen.add(cell)
        while stack:
            cur = stack.pop()
            comp.add(cur)
            x, y = cur
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                n = (nx, ny)
                if n in free and n not in seen:
                    seen.add(n)
                    stack.append(n)
        components.append(comp)
    if components:
        keep = None
        for comp in components:
            if start in comp:
                keep = comp
                break
        if keep is None:
            keep = max(components, key=len)
        walls |= (free - keep)
        walls.discard(start)

    free = {(x, y) for x in range(width) for y in range(height) if (x, y) not in walls}
    dirt = _scatter_dirt(free, start, dirt_density, rng)
    return GridWorld(width, height, obstacles=walls, dirt=dirt, start=start)


def create_floor_plan(
    width: int = 30,
    height: int = 20,
    min_room: int = 5,
    dirt_density: float = 0.2,
    seed: int = 42,
) -> GridWorld:
    rng = random.Random(seed)
    start = (0, 0)
    walls: Set[Cell] = {(x, y) for x in range(width) for y in range(height)}
    leaves: List[Tuple[int, int, int, int]] = [(1, 1, width - 2, height - 2)]  # x, y, w, h
    finished: List[Tuple[int, int, int, int]] = []

    while leaves:
        x, y, w, h = leaves.pop()
        can_split_h = h >= min_room * 2
        can_split_v = w >= min_room * 2
        if not can_split_h and not can_split_v:
            finished.append((x, y, w, h))
            continue
        if can_split_h and can_split_v:
            split_h = rng.random() < 0.5
        else:
            split_h = can_split_h
        if split_h:
            cut = rng.randint(min_room, h - min_room)
            leaves.append((x, y, w, cut))
            leaves.append((x, y + cut, w, h - cut))
        else:
            cut = rng.randint(min_room, w - min_room)
            leaves.append((x, y, cut, h))
            leaves.append((x + cut, y, w - cut, h))

    rooms: List[Tuple[int, int, int, int]] = []
    for x, y, w, h in finished:
        rx = x + 1 if w > 2 else x
        ry = y + 1 if h > 2 else y
        rw = max(1, w - 2)
        rh = max(1, h - 2)
        rooms.append((rx, ry, rw, rh))
        for gx in range(rx, min(width - 1, rx + rw)):
            for gy in range(ry, min(height - 1, ry + rh)):
                walls.discard((gx, gy))

    centers = [(x + w // 2, y + h // 2) for x, y, w, h in rooms]
    for i in range(1, len(centers)):
        x1, y1 = centers[i - 1]
        x2, y2 = centers[i]
        for x in range(min(x1, x2), max(x1, x2) + 1):
            walls.discard((x, y1))
        for y in range(min(y1, y2), max(y1, y2) + 1):
            walls.discard((x2, y))

    # keep start connected
    for x in range(0, min(3, width)):
        for y in range(0, min(3, height)):
            walls.discard((x, y))
    if centers:
        cx, cy = centers[0]
        for x in range(0, cx + 1):
            walls.discard((x, 0))
        for y in range(0, cy + 1):
            walls.discard((cx, y))

    walls.discard(start)
    free = {(x, y) for x in range(width) for y in range(height) if (x, y) not in walls}
    dirt = _scatter_dirt(free, start, dirt_density, rng)
    return GridWorld(width, height, obstacles=walls, dirt=dirt, start=start)


def create_warehouse(
    width: int = 30,
    height: int = 20,
    shelf_rows: int = 4,
    shelf_width: int = 2,
    aisle_width: int = 2,
    dirt_density: float = 0.2,
    seed: int = 42,
) -> GridWorld:
    rng = random.Random(seed)
    start = (0, 0)
    walls: Set[Cell] = set()

    dock_height = max(2, height // 5)
    y_cursor = dock_height + 1
    for _ in range(shelf_rows):
        for y in range(y_cursor, min(height - 1, y_cursor + shelf_width)):
            for x in range(1, width - 1):
                if rng.random() < 0.08:
                    continue
                walls.add((x, y))
        y_cursor += shelf_width + aisle_width
        if y_cursor >= height - 1:
            break

    stride = max(5, width // 5)
    for x in range(2, width - 2, stride):
        for y in range(1, height - 1):
            walls.discard((x, y))
            if x + 1 < width - 1:
                walls.discard((x + 1, y))

    for x in range(0, width):
        for y in range(0, dock_height):
            walls.discard((x, y))

    walls.discard(start)
    free = {(x, y) for x in range(width) for y in range(height) if (x, y) not in walls}
    dirt = _scatter_dirt(free, start, dirt_density, rng)
    return GridWorld(width, height, obstacles=walls, dirt=dirt, start=start)
