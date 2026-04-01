"""Unit and integration tests for the vacuum agent (plan §6)."""

from __future__ import annotations

import pytest

from vacuum_mas.frontier import compute_frontier, unknown_degree
from vacuum_mas.navigation import Navigator, NavigationError
from vacuum_mas.agent import VacuumAgent, run_agent_on_world
from vacuum_mas.simulator import (
    GridWorld,
    create_single_room,
    create_two_rooms,
    create_tree_of_rooms,
    create_l_shaped_room,
    create_maze,
)
from vacuum_mas.state import AgentState


def test_bfs_path_trivial():
    assert Navigator.bfs_path((0, 0), (0, 0), {(0, 0)}) == [(0, 0)]


def test_bfs_path_line():
    M = {(0, 0), (1, 0), (2, 0)}
    p = Navigator.bfs_path((0, 0), (2, 0), M)
    assert p == [(0, 0), (1, 0), (2, 0)]


def test_bfs_unreachable():
    M = {(0, 0), (2, 0)}
    with pytest.raises(NavigationError):
        Navigator.bfs_path((0, 0), (2, 0), M)


def test_frontier_unknown_adjacent_to_m():
    M = {(0, 0)}
    O: set = set()
    F = compute_frontier(M, O, None)
    assert (0, 1) in F and (1, 0) in F and (0, -1) in F and (-1, 0) in F


def test_frontier_respects_o():
    M = {(0, 0)}
    O = {(0, 1)}
    F = compute_frontier(M, O, None)
    assert (0, 1) not in F


def test_unknown_degree():
    M = {(0, 0)}
    O = set()
    f = (0, 1)
    assert unknown_degree(f, M, O, None) >= 1


def test_completeness_single_room_no_lookahead():
    w = create_single_room(5, 5, seed=99)
    _, m = run_agent_on_world(w, use_lookahead=False, max_steps=200_000)
    assert m["success"]
    assert m["matches_reachable"]
    assert m["dirt_remaining"] == 0


def test_completeness_single_room_lookahead():
    w = create_single_room(5, 5, seed=99)
    _, m = run_agent_on_world(w, use_lookahead=True, max_steps=200_000)
    assert m["success"]
    assert m["dirt_remaining"] == 0


def test_completeness_two_rooms():
    w = create_two_rooms(seed=7)
    _, m = run_agent_on_world(w, use_lookahead=True, max_steps=300_000)
    assert m["success"]


def test_completeness_tree():
    w = create_tree_of_rooms(seed=3)
    _, m = run_agent_on_world(w, use_lookahead=True, max_steps=400_000)
    assert m["success"]


def test_completeness_l_shaped():
    w = create_l_shaped_room(seed=11)
    _, m = run_agent_on_world(w, use_lookahead=True, max_steps=400_000)
    assert m["success"]


def test_completeness_maze():
    w = create_maze(seed=5)
    _, m = run_agent_on_world(w, use_lookahead=True, max_steps=400_000)
    assert m["success"]


def test_determinism():
    w = create_single_room(4, 4, seed=123)
    _, m1 = run_agent_on_world(w, use_lookahead=True, frontier_mode="nearest", max_steps=100_000)
    w2 = create_single_room(4, 4, seed=123)
    _, m2 = run_agent_on_world(w2, use_lookahead=True, frontier_mode="nearest", max_steps=100_000)
    assert m1["steps"] == m2["steps"]


def test_invariants_each_step_small_grid():
    w = GridWorld(3, 3, obstacles=set(), dirt=set(), start=(0, 0))
    st = AgentState(use_lookahead=False)
    agent = VacuumAgent(world=w, state=st, frontier_mode="nearest")
    for _ in range(10_000):
        if not agent.step_once():
            break
        agent.state.check_invariants()
    assert agent.success
    assert len(w.dirt) == 0


def test_frontier_modes_all_succeed():
    w = create_single_room(4, 4, seed=0)
    for mode in ("nearest", "warnsdorff", "boustrophedon"):
        w2 = create_single_room(4, 4, seed=0)
        _, m = run_agent_on_world(
            w2, use_lookahead=True, frontier_mode=mode, max_steps=150_000
        )
        assert m["success"], mode
