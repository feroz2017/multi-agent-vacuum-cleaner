# Implementation Plan: Vacuum Cleaning Agent in Python

## Context

The user has a detailed 62-page markdown document describing an "Elaborated & Optimized Solution" for a vacuum cleaning agent assignment in a discrete grid world. This document covers:

- PEAS framework and environment properties (AIMA 4th edition)
- Model-based reflex agent architecture
- Frontier-based exploration with formal correctness proofs
- Navigation algorithms (BFS on known free space)
- Coverage heuristics (Warnsdorff, boustrophedon)
- DFS narrative over tree of rooms
- Complete agent loop with 6 ordered phases
- Complexity analysis and alternatives

The user wants to convert this professional reference document into a working Python implementation. This will turn theoretical design into executable code that:

1. Simulates the agent's behavior in a grid world
2. Demonstrates frontier-based exploration
3. Proves correctness through termination and complete coverage
4. Supports testing with various grid layouts

## Recommended Approach

### Architecture Overview

**Three-Layer System:**

1. **Simulator Layer** (`grid_world.py`)

   - Manages the true grid world state
   - Provides sensors (dirt_here, bump)
   - Executes actuators (move_one_step, change_direction, suck)
   - Tracks ground truth vs agent knowledge
2. **Agent Layer** (`agent.py`)

   - Implements the complete agent loop with 6 phases
   - Manages internal state (position, heading, maps M and O)
   - Executes frontier exploration algorithm
   - Handles navigation and probing
3. **Support Modules** (various)

   - `frontier.py` - frontier computation and selection heuristics
   - `navigation.py` - BFS pathfinding on known subgraph
   - `visualization.py` - ASCII visualization of grid and agent progress
   - `utils.py` - common utilities (coordinate math, neighbor queries)

**Single Agent Loop** (following document 10):

- Phase 1: SENSE & CLEAN (suck if dirt detected)
- Phase 2: MAP UPDATE (apply bump percept)
- Phase 3: FRONTIER COMPUTATION & STOP CHECK
- Phase 4: FRONTIER SELECTION (choose target f and pivot p)
- Phase 5: NAVIGATION (move toward pivot p via BFS)
- Phase 6: PROBING (attempt to enter frontier f)

---

## Libraries and Justification

### Core Libraries

| Library                     | Purpose                        | Justification                                            |
| --------------------------- | ------------------------------ | -------------------------------------------------------- |
| **collections.deque** | BFS queue for pathfinding      | Efficient O(1) append/popleft; built-in, no dependency   |
| **dataclasses**       | Type-safe state representation | Clean syntax for Cell, Pose, State; Python 3.7+ standard |
| **enum.Enum**         | Direction and sensor enums     | Type-safe heading (N/E/S/W) and states                   |
| **typing**            | Type hints throughout          | Catches bugs, improves documentation                     |
| **heapq**             | Optional A* or Dijkstra        | If extending to utility-based navigation (doc 08.6)      |
| **pytest**            | Testing framework              | Industry standard; easy parametrized tests               |
| **matplotlib**        | Optional visualization         | Step-by-step grid visualization during execution         |

**No additional dependencies needed.** The core agent can be implemented in pure Python.

### Why This Stack

- **Minimal dependencies:** Only standard library for core agent; pytest/matplotlib optional for testing/visualization
- **Type-safe:** dataclasses + typing catch errors at development time
- **Educational:** Code is clean and understandable; matches textbook pseudocode from document 10.2
- **Scalable:** Easy to extend with learnable agent (doc 12.6) or other variants

---

## File Structure

```
vacuum_agent/
├── __init__.py
├── simulator.py           # GridWorld class: true environment
├── agent.py              # VacuumAgent class: main agent loop
├── state.py              # Data structures (Pose, Cell, AgentState)
├── frontier.py           # Frontier computation and selection heuristics
├── navigation.py         # BFS pathfinding
├── utils.py              # Coordinate math, neighbor queries, helpers
├── visualization.py      # ASCII grid and step visualization
├── main.py               # Example usage and test scenarios
├── test_agent.py         # Comprehensive unit tests
└── README.md             # Usage and results
```

---

## Core Components (Classes & Modules)

### 1. **state.py** — Data Structures

```python
from enum import Enum
from dataclasses import dataclass
from typing import Set, Dict, Optional, Tuple

class Direction(Enum):
    NORTH = (0, +1)
    EAST = (+1, 0)
    SOUTH = (0, -1)
    WEST = (-1, 0)

@dataclass
class Pose:
    """Agent's current position and heading."""
    x: int
    y: int
    heading: Direction

@dataclass
class Cell:
    """A single grid cell."""
    x: int
    y: int

    def __hash__(self): return hash((self.x, self.y))
    def __eq__(self, other): return self.x == other.x and self.y == other.y

class CellState(Enum):
    """Cell classification from agent's perspective."""
    UNKNOWN = "?"
    BLOCKED = "#"
    FREE_VISITED = "."
    FREE_UNVISITED = "o"  # Only with lookahead sensor

@dataclass
class AgentState:
    """Complete internal state of the agent."""
    pose: Pose
    M: Set[Cell]              # Visited free cells
    O: Set[Cell]              # Known blocked cells
    U: Set[Cell]              # Free-unvisited (lookahead sensor only)
    parent: Dict[Cell, Cell]  # Discovery tree
    use_lookahead: bool       # Sensor variant flag
```

### 2. **simulator.py** — GridWorld

```python
class GridWorld:
    """True environment simulator."""

    def __init__(self, width: int, height: int, obstacles: Set[Cell],
                 dirt_locations: Set[Cell]):
        """Initialize with known ground truth."""
        self.width = width
        self.height = height
        self.true_obstacles = obstacles
        self.true_dirt = dirt_locations.copy()  # mutable
        self.agent_pos = (0, 0)

    # Core sensors
    def dirt_here(self) -> bool:
        """Is there dirt at current cell?"""

    def bump(self, next_cell: Cell) -> bool:
        """Would moving to next_cell result in a bump?"""

    def blocked_ahead(self, pose: Pose) -> bool:
        """Lookahead sensor: is forward cell blocked? (optional)"""

    # Core actuators
    def move_one_step(self, pose: Pose) -> Pose:
        """Attempt move; return new pose (unchanged if bump)."""

    def change_direction(self, pose: Pose) -> Pose:
        """Rotate heading 90° clockwise."""

    def suck(self) -> None:
        """Remove dirt from current cell."""

    # Utilities
    def get_state_for_display(self) -> str:
        """Return ASCII representation."""
```

### 3. **frontier.py** — Frontier Management

```python
class FrontierManager:
    """Compute and manage frontier set F."""

    def compute_frontier(self, M: Set[Cell], O: Set[Cell],
                        U: Set[Cell] = None) -> Set[Cell]:
        """Compute F = unknown cells adjacent to M (or M ∪ U)."""
        F = set()
        # For each cell in M (or M ∪ U), add unknown neighbors
        return F

    def select_frontier_nearest(self, current_pos: Cell, F: Set[Cell],
                               M: Set[Cell]) -> Tuple[Cell, Cell]:
        """Select frontier f and its M-neighbor p by BFS distance."""
        # Run multi-source BFS from current_pos, label all M-cells with distance
        # Return argmin_{f in F} dist[M-neighbor(f)]
        pass

    def select_frontier_warnsdorff(self, current_pos: Cell, F: Set[Cell],
                                  M: Set[Cell], O: Set[Cell]) -> Tuple[Cell, Cell]:
        """Tie-break with Warnsdorff: prefer low unknown degree."""
        pass

    def select_frontier_boustrophedon(self, current_pos: Cell, F: Set[Cell],
                                     M: Set[Cell]) -> Tuple[Cell, Cell]:
        """Prefer row-by-row sweep over jumping rows."""
        pass

    def unknown_degree(self, f: Cell, M: Set[Cell], O: Set[Cell]) -> int:
        """Count unknown neighbors of f."""
        pass
```

### 4. **navigation.py** — Pathfinding

```python
class Navigator:
    """Navigate through known free space using BFS."""

    def bfs_path(self, start: Cell, goal: Cell, M: Set[Cell]) -> List[Cell]:
        """Find shortest path from start to goal using only M cells."""
        # Standard BFS
        # Return list of cells: [start, next_cell, ..., goal]
        pass

    def bfs_distances(self, start: Cell, M: Set[Cell]) -> Dict[Cell, int]:
        """Single multi-source BFS labeling all M cells with distance."""
        # Used by frontier selection heuristics
        pass

    def heading_from(self, from_cell: Cell, to_cell: Cell) -> Direction:
        """Determine heading needed to move from from_cell to to_cell."""
        dx = to_cell.x - from_cell.x
        dy = to_cell.y - from_cell.y
        if dy == +1: return Direction.NORTH
        if dx == +1: return Direction.EAST
        # ... etc
        pass
```

### 5. **agent.py** — VacuumAgent (Main Loop)

```python
class VacuumAgent:
    """Model-based reflex agent with frontier exploration."""

    def __init__(self, world: GridWorld, use_lookahead: bool = True):
        self.world = world
        self.state = AgentState(...)  # Initialize with start cell
        self.frontier_mgr = FrontierManager()
        self.navigator = Navigator()
        self.use_lookahead = use_lookahead
        self.step_count = 0
        self.history = []  # For visualization

    def run(self, max_steps: int = 100000) -> bool:
        """Execute main agent loop until frontier empty or max_steps."""

        while self.step_count < max_steps:
            # PHASE 1: SENSE & CLEAN
            if self.world.dirt_here():
                self.world.suck()
                self.step_count += 1
                self.history.append((self.state, "SUCK"))
                continue

            # PHASE 2: MAP UPDATE
            # (applied after previous move; bump handling)

            # PHASE 3: FRONTIER COMPUTATION & STOP CHECK
            F = self.frontier_mgr.compute_frontier(
                self.state.M, self.state.O,
                self.state.U if self.use_lookahead else None
            )

            if len(F) == 0:
                # TERMINATE
                return True  # Success

            # PHASE 4: FRONTIER SELECTION
            f, p = self.frontier_mgr.select_frontier_nearest(
                current_pos=Cell(self.state.pose.x, self.state.pose.y),
                F=F, M=self.state.M
            )

            # PHASE 5: NAVIGATION
            current_cell = Cell(self.state.pose.x, self.state.pose.y)
            if current_cell != p:
                path = self.navigator.bfs_path(current_cell, p, self.state.M)
                next_cell = path[1]
                desired_heading = self.navigator.heading_from(current_cell, next_cell)

                # Turn first if needed
                while self.state.pose.heading != desired_heading:
                    self.state.pose = self.world.change_direction(self.state.pose)
                    self.step_count += 1
                    self.history.append((self.state.copy(), "TURN"))

                # Then move
                old_pos = self.state.pose
                self.state.pose = self.world.move_one_step(self.state.pose)
                self.state.M.add(Cell(self.state.pose.x, self.state.pose.y))
                self.step_count += 1
                self.history.append((self.state.copy(), "MOVE"))
                continue

            # PHASE 6: PROBE FRONTIER
            desired_heading = self.navigator.heading_from(p, f)

            # Turn if needed
            while self.state.pose.heading != desired_heading:
                self.state.pose = self.world.change_direction(self.state.pose)
                self.step_count += 1
                self.history.append((self.state.copy(), "TURN"))

            # Probe: try to enter f
            if self.world.bump(f):
                self.state.O.add(f)
                self.history.append((self.state.copy(), "BUMP"))
            else:
                self.state.pose = self.world.move_one_step(self.state.pose)
                self.state.M.add(f)
                self.state.parent[f] = p
                self.history.append((self.state.copy(), "PROBE_ENTER"))

            self.step_count += 1

        return False  # Timeout

    def get_results(self) -> dict:
        """Return mission metrics."""
        return {
            "success": len(self.frontier_mgr.compute_frontier(...)) == 0,
            "steps": self.step_count,
            "cells_visited": len(self.state.M),
            "cells_blocked": len(self.state.O),
            "dirt_remaining": sum(1 for pos in self.state.M
                                 if self.world.true_dirt[pos]),  # pseudo
        }
```

### 6. **utils.py** — Helpers

```python
def get_neighbors(cell: Cell, include_diagonals: bool = False) -> List[Cell]:
    """Get 4-adjacent (or 8-adjacent) neighbors."""

def is_adjacent(c1: Cell, c2: Cell) -> bool:
    """Are two cells 4-adjacent?"""

def manhattan_distance(c1: Cell, c2: Cell) -> int:
    """Manhattan distance between cells."""

def rotate_cw(direction: Direction) -> Direction:
    """Rotate heading 90° clockwise."""

def get_step_delta(heading: Direction) -> Tuple[int, int]:
    """Get (dx, dy) for a step in given heading."""
```

### 7. **visualization.py** — Display

```python
def render_grid(agent_state: AgentState, true_world: GridWorld) -> str:
    """Render ASCII representation of grid."""
    # Show M as '.', O as '#', F as '~', unknown as '?'
    # Show agent position as '@'
    pass

def visualize_step_by_step(agent: VacuumAgent, grid_world: GridWorld):
    """Step through execution with printouts."""
    for i, (state, action) in enumerate(agent.history):
        print(f"\n=== Step {i}: {action} ===")
        print(render_grid(state, grid_world))
```

---

## Data Structures

### Primary Maps

| Structure        | Type                 | Purpose                         | Invariant                                             |
| ---------------- | -------------------- | ------------------------------- | ----------------------------------------------------- |
| **M**      | `Set[Cell]`        | Visited free cells              | Monotonically growing; agent.pos ∈ M always          |
| **O**      | `Set[Cell]`        | Known blocked cells             | Monotonically growing; disjoint from M                |
| **U**      | `Set[Cell]`        | Free-unvisited (lookahead only) | Subset of unknown cells; disjoint from M and O        |
| **parent** | `Dict[Cell→Cell]` | Discovery tree                  | Each cell has one parent; forms acyclic spanning tree |

### Frontier Set (Computed)

```python
F = { c ∈ Z² : c ∉ (M ∪ O)  AND  c has a 4-neighbor in M }
```

Recomputed each tick; not stored persistently (O(1) update when M or O changes).

### Pose and Position

```python
pose = (x, y, heading)  # Dead-reckoned position and direction
```

Updated atomically after each move or turn.

---

## Algorithm Implementation Details

### Frontier Exploration (Document 06)

```python
# Main loop continues while F ≠ empty

def frontier_exploration_complete_proof():
    """
    Theorem: In a finite, static, deterministic grid,
    when F = empty, M = R (reachable free set).

    Proof (contrapositive):
    - If ∃ u ∈ R \ M, then there's a path s → ... → u through free cells.
    - First cell u' ≠ u on this path that's not in M is:
      - Not blocked (on free path) → u' ∉ O
      - Not in M → unknown
      - Adjacent to a cell in M → u' ∈ F
    - Therefore F ≠ empty. QED
    """
    pass
```

### BFS Pathfinding (Document 07)

```python
def bfs_path(start, goal, M):
    """
    Find shortest path using only M cells (known free space).

    Complexity: O(|M|) per call
    Optimization: Multi-source BFS labels all M cells once.
    """
    from collections import deque

    if start == goal:
        return [start]

    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        current, path = queue.popleft()

        for neighbor in get_neighbors(current):
            if neighbor in M and neighbor not in visited:
                new_path = path + [neighbor]
                if neighbor == goal:
                    return new_path
                visited.add(neighbor)
                queue.append((neighbor, new_path))

    raise NavigationError(f"Goal {goal} unreachable from {start} in M")
```

### Frontier Selection Heuristics (Document 08)

1. **Nearest-frontier (primary)**

   - Compute BFS distances from current cell to all M cells
   - Select frontier f whose M-neighbor p has minimum distance
   - Complexity: O(|M| + |F|) with multi-source BFS
2. **Warnsdorff tie-breaker (secondary)**

   - Among nearest frontiers, prefer those with lowest unknown degree
   - unknown_degree(f) = count of unknown neighbors
   - Intuition: tight spots (corners, dead ends) should be explored while accessible
3. **Boustrophedon bias (tertiary)**

   - Prefer frontiers on current row before switching rows
   - Implement as: if any frontier on current_y, select nearest one
   - Otherwise, select frontier with closest y-coordinate
   - Systematic sweep pattern reduces cross-room transit

**Combination strategy:** Use nearest as primary; apply Warnsdorff among ties; apply boustrophedon within tied clusters.

### Six-Phase Agent Loop (Document 10)

Each tick:

1. **SENSE & CLEAN** (reflex, highest priority)

   - If dirt detected, suck and restart tick
2. **MAP UPDATE** (model maintenance)

   - Apply bump percept from previous tick
   - With lookahead: classify forward cell as free-unvisited if not blocked
3. **FRONTIER CHECK & STOP** (termination detection)

   - Compute F
   - If F = ∅, terminate (mission complete)
4. **FRONTIER SELECTION** (goal selection)

   - Choose (f, p): frontier f and its M-neighbor p
   - Apply heuristics
5. **NAVIGATION** (path execution)

   - If current ≠ p, compute BFS path to p
   - Execute one primitive action (turn or move)
   - Continue next tick
6. **PROBING** (frontier entry attempt)

   - Turn toward f if needed
   - Attempt move_one_step or receive bump
   - Update M or O accordingly

Each phase is sequential (no concurrency); priority order is fixed.

---

## Testing Strategy

### Unit Tests (pytest)

```python
# test_agent.py

def test_frontier_closure():
    """Verify F = empty when M = R."""

def test_simple_single_room():
    """Agent cleans simple 5x5 room with no obstacles."""

def test_multiple_rooms_tree():
    """Agent explores tree-of-rooms layout."""

def test_narrow_corridors():
    """Agent navigates narrow passages."""

def test_warnsdorff_preference():
    """Tight corners explored before open areas."""

def test_boustrophedon_sweep():
    """Open rooms swept row-by-row."""

def test_deadlock_free():
    """Agent never gets stuck in infinite loop."""

def test_completeness():
    """All reachable cells visited and cleaned."""

def test_determinism():
    """Same layout + same heuristics = same path."""
```

### Integration Tests

```python
# Predefined grid layouts to test:

layouts = {
    "single_room_5x5": {...},
    "L_shaped_rooms": {...},
    "narrow_corridor": {...},
    "complex_maze": {...},
    "open_grid_10x10": {...},
}

for name, layout in layouts.items():
    agent = VacuumAgent(layout)
    success = agent.run()
    metrics = agent.get_results()
    assert success, f"Failed on {name}"
    assert metrics["cells_visited"] == metrics["true_reachable_cells"]
```

### Verification Metrics

```python
def verify_correctness(agent, true_world):
    """Check hard constraints."""
    # (1) All reachable cells visited
    # (2) All visited cells clean
    # (3) Frontier empty
    # (4) No violations of monotonicity (M, O never shrink)
    # (5) All parent pointers form acyclic tree
    return all_checks_pass

def analyze_efficiency(agent):
    """Report soft metrics."""
    return {
        "steps_taken": agent.step_count,
        "cells_visited": len(agent.state.M),
        "moves": count_move_actions,
        "turns": count_turn_actions,
        "sucks": count_suck_actions,
        "frontier_selections": len(agent.history) // 6,  # approximate
    }
```

---

## Implementation Order

1. **Phase 1: Foundation (state.py, utils.py)**

   - Define Cell, Direction, Pose, AgentState
   - Implement coordinate utilities
2. **Phase 2: Simulation (simulator.py)**

   - Implement GridWorld with sensors and actuators
   - Create test grid layouts
3. **Phase 3: Core Agent (frontier.py, navigation.py)**

   - Implement frontier computation
   - Implement BFS pathfinding
   - Implement heuristic selection
4. **Phase 4: Agent Loop (agent.py)**

   - Implement 6-phase main loop
   - Integration with frontier and navigation
5. **Phase 5: Visualization (visualization.py)**

   - ASCII grid rendering
   - Step-by-step execution display
6. **Phase 6: Testing (test_agent.py)**

   - Unit tests for each component
   - Integration tests with predefined layouts
   - Metrics collection
7. **Phase 7: Documentation (main.py, README.md)**

   - Example usage
   - Results and analysis

---

## How to Verify Implementation

1. **Run basic test:** Single 5×5 room with obstacles. Verify all cells visited.
2. **Run tree-of-rooms test:** Multi-room layout. Verify F=∅ before termination.
3. **Check invariants:** After each tick, verify:

   - Agent position ∈ M
   - M ∩ O = ∅
   - F ⊆ (unknown cells adjacent to M)
   - Parent tree is acyclic
4. **Compare metrics:** For the same grid with different heuristics, verify:

   - Nearest-frontier: fewer cross-room transits
   - Warnsdorff: fewer "re-entrances" to tight spots
   - Boustrophedon: systematic row-by-row pattern visible
5. **Determinism test:** Run same layout 3 times with same seed; verify identical execution.
6. **Coverage completeness:** Generate random 20×20 grids, verify |M| = |R| at termination (100/100 trials).
7. **Step count analysis:** Plot step counts vs grid size; verify O(N) empirical behavior (document 12.1).

---

## Optional Extensions (Not Covered in Core)

- **LRTA\* learning** (doc 12.5): Track visit counts; prefer less-visited cells
- **Neural exploration** (doc 12.6): Use neural network for frontier selection
- **Utility-based navigation** (doc 8.6): Weight moves vs turns; use Dijkstra on expanded state
- **Physical robot simulation**: Add noise, wheel slip, localization
- **GUI visualization**: matplotlib real-time rendering
- **Performance profiling**: Memory and CPU analysis for large grids

---

## Summary

This implementation preserves the **correctness, clarity, and economy** of the document design:

- **Correctness**: Frontier closure theorem directly implemented; proofs as comments
- **Clarity**: Single agent loop with 6 clear phases; no hidden state or competing controllers
- **Economy**: Minimal sensors (2), bounded memory (O(N)), sparse data structures

The code will be:

- **Modular**: Each component (frontier, navigation, agent, world) is independent
- **Testable**: Comprehensive unit and integration tests with predefined layouts
- **Extensible**: Easy to add heuristics, learnable components, or robot physics
- **Educational**: Matches textbook style; comments reference AIMA and document sections