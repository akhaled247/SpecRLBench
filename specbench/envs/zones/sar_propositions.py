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


def is_entrapped_prop(prop: str) -> bool:
    """True for per-agent ``entrapped_*`` and team ``all_entrapped`` propositions."""
    return prop == "all_entrapped" or prop.startswith("entrapped_")

# When multiple team props are simultaneously true, Büchi pruning allows only one.
_TEAM_PROP_PRIORITY = ("all_entrapped", "all_surface")


def normalize_team_props(active: list[str]) -> list[str]:
    """Return ≤1 team prop (matches zero_or_one_propositions on TEAM_PROPS)."""
    active_set = [p for p in active if p in TEAM_PROPS]
    if len(active_set) <= 1:
        return active_set
    for prop in _TEAM_PROP_PRIORITY:
        if prop in active_set:
            return [prop]
    return [active_set[0]]


def pick_per_agent_prop(
    categories: set[str],
    agent_idx: int,
    costs: dict[str, float],
) -> str | None:
    """Pick at most one casualty prop for an agent (highest positive cost wins)."""
    candidates = [
        (float(costs.get(f"cost_casualtys_{category}", 0)), f"{category}_{agent_idx}")
        for category in sorted(categories)
    ]
    candidates = [(cost, prop) for cost, prop in candidates if cost > 0]
    if not candidates:
        return None
    return max(candidates)[1]


def normalize_active_propositions(
    raw_props: list[str],
    *,
    num_agents: int,
    categories: set[str],
    include_team_props: bool,
) -> list[str]:
    """Collapse runtime props to match get_possible_assignments."""
    per_agent: dict[int, list[str]] = {i: [] for i in range(num_agents)}
    team: list[str] = []
    for prop in raw_props:
        if prop in TEAM_PROPS:
            team.append(prop)
            continue
        if prop.rsplit("_", 1)[-1].isdigit():
            agent_idx = int(prop.rsplit("_", 1)[1])
            category = prop.rsplit("_", 1)[0]
            if category in categories and 0 <= agent_idx < num_agents:
                per_agent[agent_idx].append(prop)

    result: list[str] = []
    for agent_idx in range(num_agents):
        props = per_agent[agent_idx]
        if not props:
            continue
        if len(props) == 1:
            result.append(props[0])
            continue
        # Prefer entrapped over surface when both appear without cost context.
        entrapped = sorted(p for p in props if p.startswith("entrapped_"))
        surface = sorted(p for p in props if p.startswith("surface_"))
        if entrapped:
            result.append(entrapped[0])
        elif surface:
            result.append(surface[0])

    if include_team_props:
        result.extend(normalize_team_props(team))
    return sorted(result)


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


def resolve_casualty_lidar_keys_for_observer(
    prop: str,
    observer_idx: int,
    *,
    num_agents: int = 1,
    available_keys: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """Lidar keys present in one agent's obs dict for a Büchi reach/avoid prop.

    Per-agent obs only contains keys suffixed with that agent's index (see
    ``BaseTask.process_obs``). When the active subgoal names another agent's
    prop (e.g. ``surface_0`` while acting as agent 1), fall back to the
    observer's egocentric channel for the same casualty category.
    """
    if prop in TEAM_PROPS:
        keys = [resolve_casualty_lidar_key(prop, i) for i in range(num_agents)]
        if available_keys is not None:
            keys = [k for k in keys if k in available_keys]
        if keys:
            return keys
        return [resolve_casualty_lidar_key(prop, observer_idx)]

    primary = resolve_casualty_lidar_key(prop, observer_idx)
    if available_keys is None or primary in available_keys:
        return [primary]

    category, _prop_agent = prop.rsplit("_", 1)
    fallback = f"{category}_casualtys_lidar_{observer_idx}"
    if available_keys is not None and fallback in available_keys:
        return [fallback]
    return [primary]
