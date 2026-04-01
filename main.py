#!/usr/bin/env python3
"""CLI runner for single-agent and cooperative multi-agent cleaning."""

from __future__ import annotations

import argparse

from vacuum_mas.agent import VacuumAgent
from vacuum_mas.multi_agent import MultiAgentSimulation
from vacuum_mas.simulator import (
    GridWorld,
    create_single_room,
    create_two_rooms,
    create_tree_of_rooms,
    create_l_shaped_room,
    create_maze,
)
from vacuum_mas.environments import create_cave, create_floor_plan, create_warehouse


def run_layout_single(
    name: str,
    world: GridWorld,
    *,
    use_lookahead: bool,
    frontier_mode: str,
    max_steps: int,
    verbose: bool,
) -> None:
    from vacuum_mas.state import AgentState

    st = AgentState(use_lookahead=use_lookahead)
    agent = VacuumAgent(
        world=world,
        state=st,
        frontier_mode=frontier_mode,
    )
    agent.run(max_steps=max_steps)
    metrics = agent.get_results()
    print(f"\n=== {name} (single-agent) ===")
    print(f"success: {metrics['success']}")
    print(f"steps: {metrics['steps']}")
    print(f"visited: {metrics['cells_visited']} (reachable {metrics['true_reachable']})")
    print(f"dirt_remaining: {metrics['dirt_remaining']}")
    if verbose and agent.history:
        for i, (act, snap) in enumerate(agent.history[-20:]):
            base = max(0, len(agent.history) - 20)
            print(f"  [{base + i}] {act} {snap}")


def run_layout_multi(
    name: str,
    world: GridWorld,
    *,
    use_lookahead: bool,
    frontier_mode: str,
    num_agents: int,
    max_steps: int,
    verbose: bool,
) -> None:
    sim = MultiAgentSimulation(
        world=world,
        num_agents=num_agents,
        frontier_mode=frontier_mode,
        use_lookahead=use_lookahead,
    )
    for _ in range(max_steps):
        if not sim.step_all():
            break
    metrics = sim.get_results()
    print(f"\n=== {name} (multi-agent, peer-to-peer) ===")
    print(f"success: {metrics['success']}")
    print(f"ticks: {metrics['ticks']}")
    print(f"total_steps: {metrics['total_steps']}")
    print(f"visited: {metrics['coverage']} (reachable {metrics['true_reachable']})")
    print(f"dirt_remaining: {metrics['dirt_remaining']}")
    for item in metrics["agents"]:
        print(
            f"  agent {item['id']}: steps={item['steps']} dirt={item['dirt_cleaned']} "
            f"discoveries={item['cells_discovered']} phase={item['phase']} "
            f"peers={item['known_peers']} pos={item['final_pos']}"
        )
    if verbose:
        for agent in sim.agents:
            print(f"  history agent {agent.agent_id}: {agent.history[-5:]}")


def main() -> None:
    p = argparse.ArgumentParser(description="Vacuum cleaning CLI demo")
    p.add_argument(
        "--layout",
        default="all",
        choices=["all", "single", "two", "tree", "lshape", "maze",
                 "cave", "floor", "warehouse"],
    )
    p.add_argument("--no-lookahead", action="store_true")
    p.add_argument(
        "--frontier",
        default="boustrophedon",
        choices=["nearest", "warnsdorff", "boustrophedon"],
    )
    p.add_argument("--max-steps", type=int, default=500_000)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--mode", default="multi", choices=["multi", "single"])
    p.add_argument("--agents", type=int, default=3, choices=[2, 3, 4])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    layouts = {
        "single": ("single_room_5x5", create_single_room(5, 5, seed=args.seed)),
        "two": ("two_rooms", create_two_rooms(seed=args.seed)),
        "tree": ("tree_of_rooms", create_tree_of_rooms(seed=args.seed)),
        "lshape": ("l_shaped", create_l_shaped_room(seed=args.seed)),
        "maze": ("maze_7x7", create_maze(seed=args.seed)),
        "cave": ("cave_32x20", create_cave(width=32, height=20, seed=args.seed)),
        "floor": ("floor_plan_32x20", create_floor_plan(width=32, height=20, seed=args.seed)),
        "warehouse": ("warehouse_34x20", create_warehouse(width=34, height=20, seed=args.seed)),
    }

    use_lookahead = not args.no_lookahead

    def _run(name: str, w: GridWorld) -> None:
        if args.mode == "single":
            run_layout_single(
                name, w,
                use_lookahead=use_lookahead,
                frontier_mode=args.frontier,
                max_steps=args.max_steps,
                verbose=args.verbose,
            )
        else:
            run_layout_multi(
                name, w,
                use_lookahead=use_lookahead,
                frontier_mode=args.frontier,
                num_agents=args.agents,
                max_steps=args.max_steps,
                verbose=args.verbose,
            )

    if args.layout == "all":
        for key, (name, w) in layouts.items():
            _run(name, w)
    else:
        name, w = layouts[args.layout]
        _run(name, w)


if __name__ == "__main__":
    main()
