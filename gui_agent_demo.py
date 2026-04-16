#!/usr/bin/env python3
"""Pygame UI for multi-agent sim."""

from __future__ import annotations

import argparse
from typing import Dict, List

import pygame

from vacuum_mas.grid_render import (
    COLORS,
    draw_belief_map_multi,
    draw_legend,
    draw_panel_label,
    draw_stats_panel,
    draw_true_world_multi,
    init_pygame,
)
from vacuum_mas.multi_agent import MultiAgentSimulation
from vacuum_mas.simulator import (
    create_single_room,
    create_two_rooms,
    create_tree_of_rooms,
    create_l_shaped_room,
    create_maze,
)
from vacuum_mas.environments import create_cave, create_floor_plan, create_warehouse


LAYOUTS = {
    "single": ("Single Room (6x6)", lambda: create_single_room(6, 6, seed=1)),
    "two": ("Two Rooms (8x5)", lambda: create_two_rooms(seed=1)),
    "tree": ("Tree of Rooms (12x5)", lambda: create_tree_of_rooms(seed=1)),
    "lshape": ("L-Shaped Room (8x8)", lambda: create_l_shaped_room(seed=1)),
    "maze": ("Maze (7x7)", lambda: create_maze(seed=1)),
    "cave": ("Cave (32x20)", lambda: create_cave(width=32, height=20, seed=1)),
    "floor": ("Floor Plan (32x20)", lambda: create_floor_plan(width=32, height=20, seed=1)),
    "warehouse": ("Warehouse (34x20)", lambda: create_warehouse(width=34, height=20, seed=1)),
}
LAYOUT_KEYS = list(LAYOUTS.keys())
AGENT_COUNTS = [2, 3, 4]


class VacuumGUI:
    def __init__(
        self,
        layout_key: str,
        cell_size: int,
        use_lookahead: bool,
        frontier_mode: str,
        num_agents: int,
    ):
        self.cell_size = cell_size
        self.use_lookahead = use_lookahead
        self.frontier_mode = frontier_mode
        self.layout_key = layout_key
        self.num_agents = num_agents
        self.highlight_agent_idx = 0

        self.paused = True
        self.speed = 10
        self.min_speed = 1
        self.max_speed = 200

        self._init_agent()
        self._init_window()

    def _init_agent(self) -> None:
        _, factory = LAYOUTS[self.layout_key]
        self.world = factory()
        self.sim = MultiAgentSimulation(
            world=self.world,
            num_agents=self.num_agents,
            frontier_mode=self.frontier_mode,
            use_lookahead=self.use_lookahead,
        )
        self.total_dirt = len(self.world.dirt)
        self.initial_reachable = len(self.world.reachable)

    def _init_window(self) -> None:
        cs = self.cell_size
        w, h = self.world.width, self.world.height
        grid_w = w * cs
        grid_h = h * cs

        self.gap = 16
        self.stats_w = 300
        self.header_h = 32
        self.legend_h = 32
        self.controls_h = 28

        total_w = grid_w + self.gap + grid_w + self.gap + self.stats_w
        total_h = self.header_h + grid_h + self.legend_h + self.controls_h
        total_w = max(total_w, 900)

        self.window_w = total_w
        self.window_h = total_h
        self.grid_w = grid_w
        self.grid_h = grid_h

        self.screen = pygame.display.set_mode((self.window_w, self.window_h))
        pygame.display.set_caption(
            f"Multi-Agent Vacuum: {LAYOUTS[self.layout_key][0]}"
        )

        self.font_small = pygame.font.Font(None, 18)
        self.font_medium = pygame.font.Font(None, 22)
        self.font_title = pygame.font.Font(None, 26)
        self.clock = pygame.time.Clock()

    def _reset_agent(self) -> None:
        self._init_agent()
        self._init_window()

    def _switch_layout(self, key: str) -> None:
        self.layout_key = key
        self._reset_agent()

    def _toggle_lookahead(self) -> None:
        self.use_lookahead = not self.use_lookahead
        self._reset_agent()

    def _cycle_agents(self) -> None:
        idx = AGENT_COUNTS.index(self.num_agents)
        self.num_agents = AGENT_COUNTS[(idx + 1) % len(AGENT_COUNTS)]
        self._reset_agent()

    def _build_stats(self) -> Dict[str, str]:
        shared = self.sim.shared_map
        F = shared.compute_global_frontier(self.use_lookahead)
        dirt_cleaned = self.total_dirt - len(self.world.dirt)
        coverage = len(shared.M) / self.initial_reachable * 100 if self.initial_reachable else 0
        agents = self.sim.agents
        if agents:
            self.highlight_agent_idx %= len(agents)
            sel = agents[self.highlight_agent_idx]
            sel_txt = f"A{sel.agent_id} {sel.phase} {sel.last_action}"
        else:
            sel_txt = "-"

        peers_known = sum(len(a.known_agents) for a in agents)
        stats = {
            "Ticks": str(self.sim.tick_count),
            "Coverage": f"{len(shared.M)}/{self.initial_reachable} ({coverage:.0f}%)",
            "Visited (M)": str(len(shared.M)),
            "Blocked (O)": str(len(shared.O)),
            "Unvisited (U)": str(len(shared.U)),
            "Frontier (F)": str(len(F)),
            "Dirt cleaned": f"{dirt_cleaned}/{self.total_dirt}",
            "Dirt left": str(len(self.world.dirt)),
            "Agents": str(len(agents)),
            "Peer links": str(peers_known),
            "Highlighted": sel_txt,
            "Heuristic": self.frontier_mode,
            "Lookahead": "ON" if self.use_lookahead else "OFF",
            "Status": "DONE" if self.sim.terminated else ("PAUSED" if self.paused else "RUNNING"),
        }
        return stats

    def run(self) -> None:
        running = True
        step_accumulator = 0.0

        while running:
            dt = self.clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key)

            if not self.paused and not self.sim.terminated:
                step_accumulator += self.speed * dt
                steps_this_frame = int(step_accumulator)
                step_accumulator -= steps_this_frame
                for _ in range(steps_this_frame):
                    if not self.sim.step_all():
                        break

            self._draw()
            pygame.display.flip()

        pygame.quit()

    def _handle_key(self, key: int) -> bool:
        if key in (pygame.K_ESCAPE, pygame.K_q):
            return False
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_RIGHT:
            if self.paused and not self.sim.terminated:
                self.sim.step_all()
        elif key == pygame.K_UP:
            self.speed = min(self.speed * 2, self.max_speed)
        elif key == pygame.K_DOWN:
            self.speed = max(self.speed // 2, self.min_speed)
        elif key == pygame.K_r:
            self._reset_agent()
        elif key == pygame.K_n:
            self._cycle_agents()
        elif key == pygame.K_TAB:
            self.highlight_agent_idx += 1
        elif key == pygame.K_l:
            self._toggle_lookahead()
        elif key in (
            pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
            pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8,
        ):
            idx = key - pygame.K_1
            if idx < len(LAYOUT_KEYS):
                self._switch_layout(LAYOUT_KEYS[idx])
        return True

    def _draw(self) -> None:
        self.screen.fill(COLORS["bg"])
        cs = self.cell_size
        ox_left = 0
        ox_right = self.grid_w + self.gap
        oy = self.header_h

        draw_panel_label(
            self.screen, "True World", ox_left + self.grid_w // 2,
            6, self.font_title, COLORS["text"],
        )
        draw_panel_label(
            self.screen, "Shared Belief Map", ox_right + self.grid_w // 2,
            6, self.font_title, COLORS["text"],
        )

        div_x = self.grid_w + self.gap // 2
        pygame.draw.line(
            self.screen, COLORS["divider"],
            (div_x, self.header_h), (div_x, self.header_h + self.grid_h),
        )

        agent_specs = [(a.pos, a.heading, a.color) for a in self.sim.agents]
        draw_true_world_multi(self.screen, self.world, agent_specs, cs, ox_left, oy)

        view = self.sim.shared_map
        draw_belief_map_multi(
            self.screen,
            self.world,
            view.M,
            view.O,
            view.U,
            view.assigned_frontiers,
            [(a.agent_id, a.pos, a.heading, a.color) for a in self.sim.agents],
            {a.agent_id: a.current_path for a in self.sim.agents},
            {a.agent_id: a.assigned_frontier for a in self.sim.agents},
            cs,
            ox_right,
            oy,
        )

        stats_x = ox_right + self.grid_w + self.gap
        stats = self._build_stats()
        draw_stats_panel(
            self.screen, stats_x, oy, self.stats_w,
            self.grid_h, stats, self.font_small, self.font_medium,
        )

        legend_y = self.header_h + self.grid_h
        draw_legend(self.screen, legend_y, self.window_w, self.font_small)

        controls_y = legend_y + self.legend_h
        pygame.draw.rect(
            self.screen, COLORS["panel_bg"],
            (0, controls_y, self.window_w, self.controls_h),
        )
        controls_text = (
            "SPACE: Play/Pause  |  RIGHT: Step  |  UP/DOWN: Speed  |  "
            "R: Reset  |  1-8: Layout  |  N: Agents  |  "
            "TAB: Highlight  |  L: Lookahead  |  "
            f"Speed: {self.speed} ticks/s"
        )
        ctrl_surf = self.font_small.render(controls_text, True, COLORS["text_dim"])
        self.screen.blit(ctrl_surf, (8, controls_y + 7))


def main() -> None:
    p = argparse.ArgumentParser(description="Cooperative vacuum cleaning (GUI)")
    p.add_argument(
        "--layout", default="single",
        choices=list(LAYOUTS.keys()),
        help="Starting layout (switch with keys 1-8)",
    )
    p.add_argument("--cell", type=int, default=36, help="Cell size in pixels")
    p.add_argument("--no-lookahead", action="store_true")
    p.add_argument(
        "--frontier", default="boustrophedon",
        choices=["nearest", "warnsdorff", "boustrophedon"],
        help="Frontier selection heuristic",
    )
    p.add_argument("--agents", type=int, default=3, choices=AGENT_COUNTS)
    args = p.parse_args()

    init_pygame()
    gui = VacuumGUI(
        layout_key=args.layout,
        cell_size=args.cell,
        use_lookahead=not args.no_lookahead,
        frontier_mode=args.frontier,
        num_agents=args.agents,
    )
    gui.run()


if __name__ == "__main__":
    main()
