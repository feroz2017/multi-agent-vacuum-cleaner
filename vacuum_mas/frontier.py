"""
Frontier — Compute F and select (frontier_cell, pivot) pairs.

F = { c | c ∉ M ∪ O ∪ U  and  c is 4-adjacent to some cell in M ∪ U }
(When U is unused / empty, this matches classic unknown-adjacent-to-M.)

Selection: nearest BFS distance to pivot p ∈ M, then Warnsdorff (fewer unknown
neighbors of f), then boustrophedon (prefer f on same row as current).
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

from .state import Cell
from .utils import get_neighbors
from .navigation import Navigator


def compute_frontier(
    M: Set[Cell],
    O: Set[Cell],
    U: Optional[Set[Cell]] = None,
) -> Set[Cell]:
    """
    Unknown cells adjacent to known-free region (M, or M ∪ U when U is given).

    Cells in U are known free but unvisited — they are not frontiers themselves,
    but expand the region from which unknown neighbors are discovered.
    """
    base: Set[Cell] = set(M)
    if U:
        base |= U

    classified = set(M) | set(O)
    if U:
        classified |= U

    F: Set[Cell] = set()
    for cell in base:
        for n in get_neighbors(cell):
            if n not in classified:
                F.add(n)
    return F


def unknown_degree(
    f: Cell,
    M: Set[Cell],
    O: Set[Cell],
    U: Optional[Set[Cell]] = None,
) -> int:
    """Count unknown (unclassified) 4-neighbors of cell f."""
    u_set = U if U is not None else set()
    c = 0
    for n in get_neighbors(f):
        if n in M or n in O or n in u_set:
            continue
        c += 1
    return c


def pivots_for_frontier(f: Cell, M: Set[Cell]) -> List[Cell]:
    """Neighbors of f that lie in M (valid stand positions before probing f)."""
    return [p for p in get_neighbors(f) if p in M]


def frontiers_with_m_pivot(F: Set[Cell], M: Set[Cell]) -> Set[Cell]:
    """Subset of F for which some neighbor lies in M (required before probing)."""
    return {f for f in F if pivots_for_frontier(f, M)}


class FrontierManager:
    """Compute frontiers and select (f, p) with document §8 heuristics."""

    def compute_frontier(
        self,
        M: Set[Cell],
        O: Set[Cell],
        U: Optional[Set[Cell]] = None,
    ) -> Set[Cell]:
        return compute_frontier(M, O, U)

    def select_frontier_nearest(
        self,
        current_pos: Cell,
        F: Set[Cell],
        M: Set[Cell],
        O: Set[Cell],
        U: Optional[Set[Cell]] = None,
    ) -> Tuple[Cell, Cell]:
        """Primary: minimize BFS distance from current_pos to pivot p ∈ M."""
        if not F:
            raise ValueError("Frontier F is empty")

        dist = Navigator.bfs_distances(current_pos, M)
        best: Optional[Tuple[Cell, Cell]] = None
        best_key: Optional[Tuple] = None

        for f in F:
            for p in pivots_for_frontier(f, M):
                if p not in dist:
                    continue
                d = dist[p]
                key = (d, f[0], f[1], p[0], p[1])
                if best_key is None or key < best_key:
                    best_key = key
                    best = (f, p)

        if best is None:
            raise ValueError(
                "No pivot in M adjacent to any frontier cell reachable from current position"
            )
        return best

    def select_frontier_warnsdorff(
        self,
        current_pos: Cell,
        F: Set[Cell],
        M: Set[Cell],
        O: Set[Cell],
        U: Optional[Set[Cell]] = None,
    ) -> Tuple[Cell, Cell]:
        """Nearest first; tie-break by lower unknown_degree(f)."""
        if not F:
            raise ValueError("Frontier F is empty")

        dist = Navigator.bfs_distances(current_pos, M)
        best: Optional[Tuple[Cell, Cell]] = None
        best_key: Optional[Tuple] = None

        for f in F:
            wd = unknown_degree(f, M, O, U)
            for p in pivots_for_frontier(f, M):
                if p not in dist:
                    continue
                d = dist[p]
                key = (d, wd, f[0], f[1], p[0], p[1])
                if best_key is None or key < best_key:
                    best_key = key
                    best = (f, p)

        if best is None:
            raise ValueError("No valid (f, p) pair for frontier selection")
        return best

    def select_frontier_boustrophedon(
        self,
        current_pos: Cell,
        F: Set[Cell],
        M: Set[Cell],
        O: Set[Cell],
        U: Optional[Set[Cell]] = None,
    ) -> Tuple[Cell, Cell]:
        """
        Nearest + Warnsdorff, then prefer frontier on same y as agent (§8.4).
        """
        if not F:
            raise ValueError("Frontier F is empty")

        cy = current_pos[1]
        dist = Navigator.bfs_distances(current_pos, M)
        best: Optional[Tuple[Cell, Cell]] = None
        best_key: Optional[Tuple] = None

        for f in F:
            wd = unknown_degree(f, M, O, U)
            row_penalty = 0 if f[1] == cy else 1
            for p in pivots_for_frontier(f, M):
                if p not in dist:
                    continue
                d = dist[p]
                key = (d, wd, row_penalty, abs(f[1] - cy), f[0], f[1], p[0], p[1])
                if best_key is None or key < best_key:
                    best_key = key
                    best = (f, p)

        if best is None:
            raise ValueError("No valid (f, p) pair for frontier selection")
        return best

    def select_frontier(
        self,
        mode: str,
        current_pos: Cell,
        F: Set[Cell],
        M: Set[Cell],
        O: Set[Cell],
        U: Optional[Set[Cell]] = None,
    ) -> Tuple[Cell, Cell]:
        mode = mode.lower()
        if mode == "nearest":
            return self.select_frontier_nearest(current_pos, F, M, O, U)
        if mode == "warnsdorff":
            return self.select_frontier_warnsdorff(current_pos, F, M, O, U)
        if mode in ("boustrophedon", "boustro", "combined"):
            return self.select_frontier_boustrophedon(current_pos, F, M, O, U)
        raise ValueError(f"Unknown frontier mode: {mode}")
