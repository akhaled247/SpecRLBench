"""Team-level SAR proposition helpers shared by wrappers and GenZ seq_wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

from safety_gymnasium.tasks.safe_multi_agent.utils.sar_utils import (
    all_entrapped_casualties_rescued,
    all_surface_casualties_rescued,
)

if TYPE_CHECKING:
    from safety_gymnasium.tasks.safe_multi_agent.bases.base_task import BaseTask

TEAM_PROPS = ("all_entrapped", "all_surface")


def should_expose_team_props(task: BaseTask) -> bool:
    """Expose team props when more than one casualty of either type exists."""
    total_surface = 0
    total_entrapped = 0
    if hasattr(task, "surface_casualtys"):
        total_surface = task.surface_casualtys.num
    if hasattr(task, "entrapped_casualtys"):
        total_entrapped = task.entrapped_casualtys.num
    return total_surface > 1 or total_entrapped > 1


def team_active_props(task: BaseTask) -> list[str]:
    """Return team props that are currently satisfied."""
    if not should_expose_team_props(task):
        return []
    active: list[str] = []
    if all_entrapped_casualties_rescued(task):
        active.append("all_entrapped")
    if all_surface_casualties_rescued(task):
        active.append("all_surface")
    return active


def resolve_casualty_lidar_key(prop: str, agent_idx: int = 0) -> str:
    """Map a proposition name to the per-agent casualty lidar observation key."""
    if prop == "all_entrapped":
        return f"entrapped_casualtys_lidar_{agent_idx}"
    if prop == "all_surface":
        return f"surface_casualtys_lidar_{agent_idx}"
    category, idx = prop.rsplit("_", 1)
    return f"{category}_casualtys_lidar_{idx}"


def resolve_casualty_lidar_keys(
    prop: str,
    agent_idx: int = 0,
    num_agents: int = 1,
) -> list[str]:
    """Return lidar keys to max-pool for a proposition (team props span all agents)."""
    if prop in TEAM_PROPS:
        return [resolve_casualty_lidar_key(prop, i) for i in range(num_agents)]
    return [resolve_casualty_lidar_key(prop, agent_idx)]
