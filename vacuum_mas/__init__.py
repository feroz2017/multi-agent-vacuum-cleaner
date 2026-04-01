"""Multi-agent cooperative vacuum cleaning (discrete grid, peer-to-peer coordination)."""

__version__ = "1.0.0"

from .agent import VacuumAgent, run_agent_on_world
from .multi_agent import MultiAgentSimulation
from .simulator import GridWorld

__all__ = [
    "__version__",
    "VacuumAgent",
    "run_agent_on_world",
    "GridWorld",
    "MultiAgentSimulation",
]
