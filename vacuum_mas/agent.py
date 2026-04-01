"""
Agent — Model-based reflex vacuum with frontier exploration (Document §10).

Phases per control tick (one primitive action per `run` iteration):
  1. Sense & clean — suck if dirt_here
  2. Map update — lookahead: blocked_ahead → O; else forward → U (if not in M∪O)
  3. Frontier F; if empty and known-free-unvisited U nonempty → visit U (like probe)
  4. If F empty and U empty → terminate (success)
  5. Select (f, p), navigate toward p (one turn or one move)
  6. At p → probe into f (one turn or one move / bump)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .state import AgentState, Direction, Cell
from .simulator import GridWorld
from .frontier import FrontierManager, compute_frontier, frontiers_with_m_pivot
from .navigation import Navigator
from .utils import get_neighbors, heading_from


@dataclass
class VacuumAgent:
    world: GridWorld
    state: AgentState = field(default_factory=AgentState)
    frontier_mode: str = "boustrophedon"
    step_count: int = 0
    history: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)
    terminated: bool = False
    success: bool = False

    def __post_init__(self) -> None:
        self.frontier_mgr = FrontierManager()
        sx, sy = self.world.start
        self.state.x, self.state.y = sx, sy
        self.state.heading = Direction.NORTH
        self.state.M = {self.world.start}
        self.state.O = set()
        self.state.U = set()
        self.state.parent = {self.world.start: None}

    def _snapshot(self) -> Dict[str, Any]:
        return {
            "x": self.state.x,
            "y": self.state.y,
            "heading": self.state.heading.name,
            "M_size": len(self.state.M),
            "O_size": len(self.state.O),
            "U_size": len(self.state.U),
        }

    def _record(self, action: str) -> None:
        self.history.append((action, self._snapshot()))

    def _lookahead_update(self) -> None:
        if not self.state.use_lookahead:
            return
        forward = self.state.forward_cell()
        if self.world.blocked_ahead(self.state.pos, self.state.heading):
            self.state.classify_as_blocked(forward)
        else:
            if forward not in self.state.M and forward not in self.state.O:
                self.state.classify_as_free_unvisited(forward)

    def _mission_complete(self) -> bool:
        F_all = compute_frontier(
            self.state.M,
            self.state.O,
            self.state.U if self.state.use_lookahead else None,
        )
        if F_all:
            return False
        if self.state.use_lookahead and self.state.U:
            return False
        if self.world.dirt_here(self.state.pos):
            return False
        return True

    def _select_u_target(self) -> Tuple[Cell, Cell]:
        """Pick (u, p) with u ∈ U and p ∈ M adjacent to u; minimize BFS dist to p."""
        dist = Navigator.bfs_distances(self.state.pos, self.state.M)
        best_pair: Optional[Tuple[Cell, Cell]] = None
        best_key: Optional[Tuple[int, int, int, int, int, int]] = None
        for u in list(self.state.U):
            for p in get_neighbors(u):
                if p not in self.state.M:
                    continue
                d = dist.get(p)
                if d is None:
                    continue
                key = (d, abs(u[1] - self.state.pos[1]), u[0], u[1], p[0], p[1])
                if best_key is None or key < best_key:
                    best_key = key
                    best_pair = (u, p)
        if best_pair is None:
            raise RuntimeError("Invariant broken: U nonempty but no M-adjacent pivot")
        return best_pair

    def _one_step_navigate(self, goal_cell: Cell) -> None:
        """One primitive toward goal_cell ∈ M (must be reachable in M)."""
        if self.state.pos == goal_cell:
            return
        path = Navigator.bfs_path(self.state.pos, goal_cell, self.state.M)
        nxt = path[1]
        want = heading_from(self.state.pos, nxt)
        if self.state.heading != want:
            self.state.heading = self.world.rotate_cw(self.state.heading)
            self.step_count += 1
            self._record("TURN")
            return
        new_pos, bumped = self.world.try_move(self.state.pos, self.state.heading)
        if bumped:
            raise RuntimeError(
                f"Unexpected bump navigating in M: {self.state.pos} -> {nxt}"
            )
        self.state.pos = new_pos
        self.state.classify_as_free(new_pos)
        self.step_count += 1
        self._record("MOVE")

    def _one_step_probe(self, f: Cell, p: Cell) -> None:
        """At pivot p, one primitive toward frontier f."""
        assert self.state.pos == p
        want = heading_from(p, f)
        if self.state.heading != want:
            self.state.heading = self.world.rotate_cw(self.state.heading)
            self.step_count += 1
            self._record("TURN_PROBE")
            return
        new_pos, bumped = self.world.try_move(self.state.pos, self.state.heading)
        if bumped:
            self.state.classify_as_blocked(f)
            self.step_count += 1
            self._record("BUMP")
            return
        self.state.pos = new_pos
        self.state.classify_as_free(f)
        self.state.parent[f] = p
        self.step_count += 1
        self._record("PROBE")

    def _one_step_enter_known_free(self, u: Cell, p: Cell) -> None:
        """Enter cell u ∈ U from adjacent p ∈ M (lookahead: guaranteed free)."""
        assert self.state.pos == p
        want = heading_from(p, u)
        if self.state.heading != want:
            self.state.heading = self.world.rotate_cw(self.state.heading)
            self.step_count += 1
            self._record("TURN_U")
            return
        new_pos, bumped = self.world.try_move(self.state.pos, self.state.heading)
        if bumped:
            raise RuntimeError(f"Lookahead said free but bump entering {u}")
        self.state.pos = new_pos
        self.state.classify_as_free(u)
        self.state.parent[u] = p
        self.step_count += 1
        self._record("ENTER_U")

    def step_once(self) -> bool:
        """
        Execute one primitive action. Returns True if agent should continue, False if done.
        """
        if self.terminated:
            return False
        self.state.check_invariants()

        if self.world.dirt_here(self.state.pos):
            self.world.suck(self.state.pos)
            self.step_count += 1
            self._record("SUCK")
            return True

        self._lookahead_update()

        F_all = self.frontier_mgr.compute_frontier(
            self.state.M,
            self.state.O,
            self.state.U if self.state.use_lookahead else None,
        )
        F = frontiers_with_m_pivot(F_all, self.state.M)

        def try_visit_u() -> bool:
            u, p = self._select_u_target()
            if self.state.pos != p:
                self._one_step_navigate(p)
                return True
            self._one_step_enter_known_free(u, p)
            return True

        if not F:
            if self.state.use_lookahead and self.state.U:
                return try_visit_u()
            if not F_all:
                self.terminated = True
                self.success = (not self.world.dirt_here(self.state.pos)) and (
                    len(self.world.dirt) == 0
                )
                self._record("STOP")
                return False
            raise RuntimeError(
                "Frontier cells exist but none are adjacent to M; "
                "enable lookahead or fix map logic."
            )

        f, p = self.frontier_mgr.select_frontier(
            self.frontier_mode,
            self.state.pos,
            F,
            self.state.M,
            self.state.O,
            self.state.U if self.state.use_lookahead else None,
        )

        if self.state.pos != p:
            self._one_step_navigate(p)
            return True

        self._one_step_probe(f, p)
        return True

    def run(self, max_steps: int = 500_000) -> bool:
        """Run until mission complete or max_steps. Returns success flag."""
        self.terminated = False
        self.success = False
        while True:
            if self.step_count >= max_steps:
                if not self.terminated:
                    self.terminated = True
                    self.success = False
                    self._record("TIMEOUT")
                break
            if not self.step_once():
                break
        return self.success

    def get_results(self) -> Dict[str, Any]:
        F = compute_frontier(
            self.state.M,
            self.state.O,
            self.state.U if self.state.use_lookahead else None,
        )
        reachable = self.world.reachable
        return {
            "success": self.success
            and len(F) == 0
            and (not self.state.use_lookahead or len(self.state.U) == 0)
            and self.state.M == reachable
            and len(self.world.dirt) == 0,
            "terminated": self.terminated,
            "steps": self.step_count,
            "cells_visited": len(self.state.M),
            "cells_blocked_known": len(self.state.O),
            "dirt_remaining": len(self.world.dirt),
            "frontier_empty": len(F) == 0,
            "U_remaining": len(self.state.U),
            "matches_reachable": self.state.M == reachable,
            "true_reachable": len(reachable),
        }


def run_agent_on_world(
    world: GridWorld,
    *,
    use_lookahead: bool = True,
    frontier_mode: str = "boustrophedon",
    max_steps: int = 500_000,
) -> Tuple[VacuumAgent, Dict[str, Any]]:
    st = AgentState(use_lookahead=use_lookahead)
    agent = VacuumAgent(world=world, state=st, frontier_mode=frontier_mode)
    agent.run(max_steps=max_steps)
    return agent, agent.get_results()
