"""Multi-agent vacuum: message bus, local frames, HELLO/MAP/ASSIGN/DONE/ACTIVE."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .state import AgentState, Cell, Direction
from .simulator import GridWorld
from .frontier import FrontierManager, compute_frontier, frontiers_with_m_pivot
from .navigation import Navigator, NavigationError
from .utils import get_neighbors, heading_from, manhattan_distance


AGENT_COLORS: List[Tuple[int, int, int]] = [
    (40, 160, 220),   # Blue
    (220, 80, 60),    # Red
    (80, 200, 80),    # Green
    (230, 170, 40),   # Gold
    (180, 80, 220),   # Purple
    (40, 200, 180),   # Teal
    (220, 120, 180),  # Pink
    (160, 200, 40),   # Lime
]


class MessageType(Enum):
    HELLO = "HELLO"
    MAP = "MAP"
    ASSIGN = "ASSIGN"
    DONE = "DONE"
    ACTIVE = "ACTIVE"


@dataclass
class Message:
    msg_type: MessageType
    sender_id: int
    tick: int
    payload: Dict[str, Any] = field(default_factory=dict)


class MessageBus:
    """Per-agent deques; send pushes to recipient inbox."""

    def __init__(self) -> None:
        self._inboxes: Dict[int, deque] = {}

    def register(self, agent_id: int) -> None:
        self._inboxes[agent_id] = deque()

    def send(self, msg: Message, receiver_id: int) -> None:
        if receiver_id in self._inboxes:
            self._inboxes[receiver_id].append(msg)

    def broadcast(self, msg: Message) -> None:
        for aid, inbox in self._inboxes.items():
            if aid != msg.sender_id:
                inbox.append(msg)

    def receive_all(self, agent_id: int) -> List[Message]:
        inbox = self._inboxes.get(agent_id, deque())
        msgs = list(inbox)
        inbox.clear()
        return msgs


@dataclass
class CoordinateFrame:
    """Maps absolute grid coords to/from this agent's local origin."""
    origin_x: int
    origin_y: int

    def to_local(self, abs_cell: Cell) -> Cell:
        return (abs_cell[0] - self.origin_x, abs_cell[1] - self.origin_y)

    def to_absolute(self, local_cell: Cell) -> Cell:
        return (local_cell[0] + self.origin_x, local_cell[1] + self.origin_y)


def compute_frame_transform(
    my_local: Cell,
    their_local: Cell,
    offset_me_to_them: Tuple[int, int] = (0, 0),
) -> Tuple[int, int]:
    """Delta to add to peer-local cells to express them in my local frame."""
    dx = my_local[0] + offset_me_to_them[0] - their_local[0]
    dy = my_local[1] + offset_me_to_them[1] - their_local[1]
    return (dx, dy)


def translate_cells(cells: Set[Cell], t: Tuple[int, int]) -> Set[Cell]:
    dx, dy = t
    return {(x + dx, y + dy) for x, y in cells}


def translate_parent(
    parent: Dict[Cell, Optional[Cell]], t: Tuple[int, int]
) -> Dict[Cell, Optional[Cell]]:
    dx, dy = t
    out: Dict[Cell, Optional[Cell]] = {}
    for cell, par in parent.items():
        new_cell = (cell[0] + dx, cell[1] + dy)
        new_par = (par[0] + dx, par[1] + dy) if par else None
        out[new_cell] = new_par
    return out


def _reverse(t: Tuple[int, int]) -> Tuple[int, int]:
    return (-t[0], -t[1])


def compute_start_positions(world: GridWorld, num_agents: int) -> List[Cell]:
    if num_agents <= 0:
        return []
    free = sorted(world.reachable)
    starts: List[Cell] = [world.start]
    unused = [c for c in free if c != world.start]
    while len(starts) < num_agents and unused:
        best = max(unused, key=lambda c: min(manhattan_distance(c, s) for s in starts))
        starts.append(best)
        unused.remove(best)
    return starts


class AutonomousAgent:
    """One robot: local M/O/U, peers, phases exploring / coordinated / done."""

    MAP_SHARE_INTERVAL = 15

    def __init__(
        self,
        agent_id: int,
        world: GridWorld,
        abs_start: Cell,
        color: Tuple[int, int, int],
        message_bus: MessageBus,
        frontier_mode: str = "boustrophedon",
        use_lookahead: bool = True,
    ) -> None:
        self.agent_id = agent_id
        self.world = world
        self.color = color
        self.message_bus = message_bus
        self.frontier_mode = frontier_mode
        self.use_lookahead = use_lookahead

        self.frame = CoordinateFrame(abs_start[0], abs_start[1])

        self.local_pos: Cell = (0, 0)
        self.heading: Direction = Direction.NORTH
        self.local_M: Set[Cell] = {(0, 0)}
        self.local_O: Set[Cell] = set()
        self.local_U: Set[Cell] = set()
        self.local_parent: Dict[Cell, Optional[Cell]] = {(0, 0): None}

        self.known_agents: Set[int] = set()
        self.transforms: Dict[int, Tuple[int, int]] = {}
        self.known_positions: Dict[int, Cell] = {}
        self.merged_with: Set[int] = set()
        self._pending_hello_pos: Dict[int, Cell] = {}

        self.phase: str = "exploring"
        self.assigned_region: Optional[Set[Cell]] = None
        self.done_agents: Set[int] = set()
        self.is_done: bool = False

        self._target_frontier: Optional[Cell] = None
        self._target_pivot: Optional[Cell] = None
        self.current_path: List[Cell] = []

        self.step_count: int = 0
        self.dirt_cleaned: int = 0
        self.cells_discovered: int = 0
        self.last_action: str = ""
        self.history: List[Tuple[str, Dict[str, Any]]] = []

        self.frontier_mgr = FrontierManager()

    @property
    def pos(self) -> Cell:
        return self.frame.to_absolute(self.local_pos)

    @property
    def abs_pos(self) -> Cell:
        return self.pos

    @property
    def assigned_frontier(self) -> Optional[Cell]:
        if self._target_frontier is not None:
            return self.frame.to_absolute(self._target_frontier)
        return None

    def _local_forward(self) -> Cell:
        return (self.local_pos[0] + self.heading.dx,
                self.local_pos[1] + self.heading.dy)

    def _record(self, action: str) -> None:
        self.last_action = action
        self.history.append((action, {
            "agent_id": self.agent_id,
            "local_pos": self.local_pos,
            "abs_pos": self.pos,
            "heading": self.heading.name,
            "phase": self.phase,
        }))

    def step_once(self, tick: int) -> bool:
        self._sense()
        self._process_messages(tick)
        self._detect_nearby_agents(tick)

        if tick > 0 and tick % self.MAP_SHARE_INTERVAL == 0:
            self._periodic_map_share(tick)

        if self.world.dirt_here(self.pos):
            self.world.suck_for_agent(self.agent_id, self.pos)
            self.dirt_cleaned += 1
            self.step_count += 1
            self._record("SUCK")
            return True

        if self.phase == "done":
            self._check_reactivation(tick)
            if self.phase == "done":
                self._record("IDLE_DONE")
                return False

        self._explore_step(tick)
        return True

    def _sense(self) -> None:
        if self.local_pos not in self.local_M:
            self.local_M.add(self.local_pos)
            self.local_U.discard(self.local_pos)
            self.cells_discovered += 1

        if self.use_lookahead:
            fwd_local = self._local_forward()
            if self.world.blocked_ahead(self.pos, self.heading):
                self.local_O.add(fwd_local)
                self.local_U.discard(fwd_local)
            elif fwd_local not in self.local_M and fwd_local not in self.local_O:
                self.local_U.add(fwd_local)

    def _process_messages(self, tick: int) -> None:
        for msg in self.message_bus.receive_all(self.agent_id):
            if msg.msg_type == MessageType.HELLO:
                self._handle_hello(msg, tick)
            elif msg.msg_type == MessageType.MAP:
                self._handle_map(msg)
            elif msg.msg_type == MessageType.ASSIGN:
                self._handle_assign(msg)
            elif msg.msg_type == MessageType.DONE:
                self._handle_done(msg)
            elif msg.msg_type == MessageType.ACTIVE:
                self._handle_active(msg, tick)

    def _handle_hello(self, msg: Message, tick: int) -> None:
        sid = msg.sender_id
        their_local_pos: Cell = msg.payload["local_pos"]
        offset_sender_to_me: Tuple[int, int] = msg.payload.get(
            "offset_sender_to_recipient", (0, 0)
        )
        offset_me_to_sender = (-offset_sender_to_me[0], -offset_sender_to_me[1])

        if sid not in self.known_agents:
            my_local_at_hello = self._pending_hello_pos.pop(sid, self.local_pos)
            transform = compute_frame_transform(
                my_local_at_hello, their_local_pos, offset_me_to_sender
            )
            self.known_agents.add(sid)
            self.transforms[sid] = transform
            self.known_positions[sid] = (
                their_local_pos[0] + transform[0],
                their_local_pos[1] + transform[1],
            )

            self.message_bus.send(
                Message(MessageType.HELLO, self.agent_id, tick, {
                    "local_pos": self.local_pos,
                    "offset_sender_to_recipient": offset_me_to_sender,
                }),
                sid,
            )

            self._send_map_to(sid, tick)

    def _handle_map(self, msg: Message) -> None:
        sid = msg.sender_id
        if sid not in self.transforms:
            return

        remote_M: Set[Cell] = msg.payload["M"]
        remote_O: Set[Cell] = msg.payload["O"]
        remote_U: Set[Cell] = msg.payload["U"]

        self.local_M |= remote_M
        self.local_O |= remote_O
        self.local_U |= remote_U
        self.local_U -= self.local_M
        self.local_U -= self.local_O

        sender_pos = msg.payload.get("sender_local_pos")
        if sender_pos:
            self.known_positions[sid] = sender_pos

        self.merged_with.add(sid)
        if self.phase == "exploring":
            self.phase = "coordinated"

        group = {self.agent_id} | self.known_agents
        if min(group) == self.agent_id:
            self._compute_and_send_assignments(msg.tick)

    def _handle_assign(self, msg: Message) -> None:
        region = msg.payload.get("region")
        if region is not None:
            self.assigned_region = set(region)
            if self.phase == "done":
                if self.assigned_region:
                    self.phase = "coordinated"
                    self.is_done = False

    def _handle_done(self, msg: Message) -> None:
        self.done_agents.add(msg.sender_id)

    def _handle_active(self, msg: Message, tick: int) -> None:
        self.done_agents.discard(msg.sender_id)
        if self.phase == "done":
            self.phase = "coordinated" if self.merged_with else "exploring"
            self.is_done = False

    def _detect_nearby_agents(self, tick: int) -> None:
        nearby = self.world.nearby_agents_with_offset(self.pos, self.agent_id)
        for other_id, offset_me_to_them in nearby:
            if other_id not in self.known_agents and other_id not in self._pending_hello_pos:
                self._pending_hello_pos[other_id] = self.local_pos
                self.message_bus.send(
                    Message(MessageType.HELLO, self.agent_id, tick, {
                        "local_pos": self.local_pos,
                        "offset_sender_to_recipient": offset_me_to_them,
                    }),
                    other_id,
                )

    def _send_map_to(self, target_id: int, tick: int) -> None:
        rev = _reverse(self.transforms[target_id])
        self.message_bus.send(
            Message(MessageType.MAP, self.agent_id, tick, {
                "M": translate_cells(self.local_M, rev),
                "O": translate_cells(self.local_O, rev),
                "U": translate_cells(self.local_U, rev),
                "sender_local_pos": (
                    self.local_pos[0] + rev[0],
                    self.local_pos[1] + rev[1],
                ),
            }),
            target_id,
        )

    def _periodic_map_share(self, tick: int) -> None:
        for aid in self.known_agents:
            self._send_map_to(aid, tick)

    def _compute_and_send_assignments(self, tick: int) -> None:
        F_all = compute_frontier(
            self.local_M, self.local_O,
            self.local_U if self.use_lookahead else None,
        )
        F = frontiers_with_m_pivot(F_all, self.local_M)
        if not F:
            return

        agent_positions = {self.agent_id: self.local_pos}
        for aid in self.known_agents:
            if aid in self.known_positions:
                agent_positions[aid] = self.known_positions[aid]

        dist_maps: Dict[int, Dict[Cell, int]] = {}
        for aid, apos in agent_positions.items():
            if apos in self.local_M:
                dist_maps[aid] = Navigator.bfs_distances(apos, self.local_M)

        assignment: Dict[int, Set[Cell]] = {aid: set() for aid in agent_positions}
        for f in F:
            best_aid: Optional[int] = None
            best_d = float("inf")
            for aid, dmap in dist_maps.items():
                for p in get_neighbors(f):
                    if p not in dmap:
                        continue
                    d = dmap[p]
                    if d < best_d or (d == best_d and (best_aid is None or aid < best_aid)):
                        best_d = d
                        best_aid = aid
            if best_aid is not None:
                assignment[best_aid].add(f)

        self.assigned_region = assignment.get(self.agent_id, set())

        for aid in self.known_agents:
            region_my_frame = assignment.get(aid, set())
            rev = _reverse(self.transforms[aid])
            region_their_frame = translate_cells(region_my_frame, rev)
            self.message_bus.send(
                Message(MessageType.ASSIGN, self.agent_id, tick,
                        {"region": region_their_frame}),
                aid,
            )

    def _explore_step(self, tick: int) -> None:
        F_all = compute_frontier(
            self.local_M, self.local_O,
            self.local_U if self.use_lookahead else None,
        )
        F = frontiers_with_m_pivot(F_all, self.local_M)

        if self.phase == "coordinated" and self.assigned_region:
            F_filtered = {f for f in F if f in self.assigned_region}
            if F_filtered:
                F = F_filtered

        if not F:
            if self.use_lookahead and self.local_U:
                self._target_nearest_u()
                if self._target_frontier is not None:
                    self._do_navigate_or_probe()
                    return
            if not F_all:
                self._enter_done(tick)
                return
            self._record("IDLE")
            return

        if not self._has_valid_target(F):
            self._select_target(F)
        if self._target_frontier is None:
            self._record("IDLE")
            return

        self._do_navigate_or_probe()

    def _has_valid_target(self, F: Set[Cell]) -> bool:
        f = self._target_frontier
        p = self._target_pivot
        if f is None or p is None:
            return False
        if f in self.local_M or f in self.local_O:
            return False
        if p not in self.local_M:
            return False
        return True

    def _select_target(self, F: Set[Cell]) -> None:
        try:
            f, p = self.frontier_mgr.select_frontier(
                self.frontier_mode, self.local_pos, F,
                self.local_M, self.local_O,
                self.local_U if self.use_lookahead else None,
            )
            self._target_frontier = f
            self._target_pivot = p
            self._update_render_path()
        except ValueError:
            self._target_frontier = None
            self._target_pivot = None
            self.current_path = []

    def _target_nearest_u(self) -> None:
        dist = Navigator.bfs_distances(self.local_pos, self.local_M)
        best_u: Optional[Cell] = None
        best_p: Optional[Cell] = None
        best_d = float("inf")
        for u in sorted(self.local_U):
            for p in get_neighbors(u):
                if p not in self.local_M or p not in dist:
                    continue
                if dist[p] < best_d:
                    best_d = dist[p]
                    best_u = u
                    best_p = p
        if best_u and best_p:
            self._target_frontier = best_u
            self._target_pivot = best_p
            self._update_render_path()
        else:
            self._target_frontier = None
            self._target_pivot = None
            self.current_path = []

    def _update_render_path(self) -> None:
        f = self._target_frontier
        p = self._target_pivot
        if f is None or p is None:
            self.current_path = []
            return
        try:
            if self.local_pos != p:
                local_path = Navigator.bfs_path(self.local_pos, p, self.local_M)
                local_path.append(f)
            else:
                local_path = [self.local_pos, f]
            self.current_path = [self.frame.to_absolute(c) for c in local_path]
        except NavigationError:
            self.current_path = []

    def _do_navigate_or_probe(self) -> None:
        f = self._target_frontier
        p = self._target_pivot
        if f is None or p is None:
            self._record("IDLE")
            return
        if f in self.local_M or f in self.local_O:
            self._target_frontier = None
            self._target_pivot = None
            self._record("IDLE")
            return
        if p not in self.local_M:
            self._target_frontier = None
            self._target_pivot = None
            self._record("IDLE")
            return
        if self.local_pos != p:
            self._navigate_one_step(p)
        else:
            self._probe_one_step(f, p)

    def _turn_toward(self, want: Direction) -> None:
        turns_cw = self.heading.turns_to(want)
        if turns_cw <= 2:
            self.heading = self.world.rotate_cw_for_agent(self.agent_id, self.heading)
        else:
            self.heading = self.world.rotate_ccw_for_agent(self.agent_id, self.heading)

    def _navigate_one_step(self, goal: Cell) -> None:
        try:
            path = Navigator.bfs_path(self.local_pos, goal, self.local_M)
        except NavigationError:
            self._target_frontier = None
            self._target_pivot = None
            self._record("IDLE")
            return

        nxt = path[1]
        want = heading_from(self.local_pos, nxt)
        if self.heading != want:
            self._turn_toward(want)
            self.step_count += 1
            self._record("TURN")
            return

        abs_nxt = self.frame.to_absolute(nxt)
        if self.world.is_cell_occupied_by_agent(abs_nxt, self.agent_id):
            self.step_count += 1
            self._record("WAIT")
            return

        old_local = self.local_pos
        new_abs, bumped = self.world.try_move(self.pos, self.heading,
                                              agent_id=self.agent_id)
        if bumped:
            self.local_O.add(nxt)
            self.local_U.discard(nxt)
            self.step_count += 1
            self._record("BUMP")
        else:
            self.local_pos = nxt
            self.local_M.add(nxt)
            self.local_U.discard(nxt)
            self.local_parent.setdefault(nxt, old_local)
            self.cells_discovered += 1
            self.step_count += 1
            self._record("MOVE")

    def _probe_one_step(self, f: Cell, p: Cell) -> None:
        want = heading_from(p, f)
        if self.heading != want:
            self._turn_toward(want)
            self.step_count += 1
            self._record("TURN_PROBE")
            return

        abs_f = self.frame.to_absolute(f)
        if self.world.is_cell_occupied_by_agent(abs_f, self.agent_id):
            self.step_count += 1
            self._record("WAIT")
            return

        new_abs, bumped = self.world.try_move(self.pos, self.heading,
                                              agent_id=self.agent_id)
        if bumped:
            self.local_O.add(f)
            self.local_U.discard(f)
            self.step_count += 1
            self._record("BUMP")
        else:
            self.local_pos = f
            self.local_M.add(f)
            self.local_U.discard(f)
            self.local_parent.setdefault(f, p)
            self.cells_discovered += 1
            self.step_count += 1
            self._record("PROBE")

        self._target_frontier = None
        self._target_pivot = None
        self.current_path = []

    def _enter_done(self, tick: int) -> None:
        if self.phase != "done":
            self.phase = "done"
            self.is_done = True
            self.message_bus.broadcast(
                Message(MessageType.DONE, self.agent_id, tick)
            )
        self._record("IDLE_DONE")

    def _check_reactivation(self, tick: int) -> None:
        F_all = compute_frontier(
            self.local_M, self.local_O,
            self.local_U if self.use_lookahead else None,
        )
        if F_all or (self.use_lookahead and self.local_U):
            self.phase = "coordinated" if self.merged_with else "exploring"
            self.is_done = False
            self.done_agents.discard(self.agent_id)
            self.message_bus.broadcast(
                Message(MessageType.ACTIVE, self.agent_id, tick)
            )


@dataclass
class _GlobalView:
    M: Set[Cell]
    O: Set[Cell]
    U: Set[Cell]
    assigned_frontiers: Dict[int, Set[Cell]]
    agent_positions: Dict[int, Cell]

    def compute_global_frontier(self, use_lookahead: bool) -> Set[Cell]:
        F = compute_frontier(self.M, self.O, self.U if use_lookahead else None)
        return frontiers_with_m_pivot(F, self.M)


class MultiAgentSimulation:
    """Steps all agents; termination when everyone is done and peers agree."""

    def __init__(
        self,
        world: GridWorld,
        num_agents: int = 3,
        frontier_mode: str = "boustrophedon",
        use_lookahead: bool = True,
    ) -> None:
        self.world = world
        self.use_lookahead = use_lookahead
        self.message_bus = MessageBus()

        starts = compute_start_positions(world, num_agents)
        self.agents: List[AutonomousAgent] = []
        for i, start in enumerate(starts):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            agent = AutonomousAgent(
                agent_id=i,
                world=world,
                abs_start=start,
                color=color,
                message_bus=self.message_bus,
                frontier_mode=frontier_mode,
                use_lookahead=use_lookahead,
            )
            self.agents.append(agent)
            self.message_bus.register(i)
            world.register_agent(i, start)

        self.tick_count = 0
        self.terminated = False
        self.success = False

    def step_all(self) -> bool:
        if self.terminated:
            return False

        for agent in self.agents:
            agent.step_once(self.tick_count)

        self.tick_count += 1

        if all(a.is_done for a in self.agents):
            all_settled = True
            for a in self.agents:
                if a.known_agents and not (a.known_agents <= a.done_agents):
                    all_settled = False
                    break
            if all_settled:
                self.terminated = True
                self.success = len(self.world.dirt) == 0
                return False

        return True

    @property
    def shared_map(self) -> _GlobalView:
        return self._compute_global_view()

    def _compute_global_view(self) -> _GlobalView:
        gM: Set[Cell] = set()
        gO: Set[Cell] = set()
        gU: Set[Cell] = set()
        assigned: Dict[int, Set[Cell]] = {}

        for a in self.agents:
            for c in a.local_M:
                gM.add(a.frame.to_absolute(c))
            for c in a.local_O:
                gO.add(a.frame.to_absolute(c))
            for c in a.local_U:
                ac = a.frame.to_absolute(c)
                if ac not in gM and ac not in gO:
                    gU.add(ac)
            af: Set[Cell] = set()
            if a.assigned_frontier is not None:
                af.add(a.assigned_frontier)
            assigned[a.agent_id] = af

        gU -= gM
        gU -= gO

        return _GlobalView(
            gM, gO, gU, assigned,
            {a.agent_id: a.pos for a in self.agents},
        )

    def get_results(self) -> Dict[str, Any]:
        view = self._compute_global_view()
        F = compute_frontier(view.M, view.O, view.U)
        reachable = self.world.reachable
        total_steps = sum(a.step_count for a in self.agents)

        return {
            "success": (
                self.terminated
                and view.M == reachable
                and len(self.world.dirt) == 0
            ),
            "terminated": self.terminated,
            "ticks": self.tick_count,
            "total_steps": total_steps,
            "total_dirt_cleaned": sum(a.dirt_cleaned for a in self.agents),
            "coverage": len(view.M),
            "true_reachable": len(reachable),
            "matches_reachable": view.M == reachable,
            "frontier_remaining": len(F),
            "dirt_remaining": len(self.world.dirt),
            "agents": [
                {
                    "id": a.agent_id,
                    "steps": a.step_count,
                    "dirt_cleaned": a.dirt_cleaned,
                    "cells_discovered": a.cells_discovered,
                    "final_pos": a.pos,
                    "last_action": a.last_action,
                    "phase": a.phase,
                    "known_peers": len(a.known_agents),
                }
                for a in self.agents
            ],
        }


MultiAgentCoordinator = MultiAgentSimulation
