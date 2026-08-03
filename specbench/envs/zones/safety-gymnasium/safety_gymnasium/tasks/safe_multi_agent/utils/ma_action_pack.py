"""Pack per-agent action dicts into MuJoCo ``ctrl`` in live actuator order."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np

CtrlLayout = Literal["blocked", "interleaved"]


def actuator_names(model: Any) -> list[str]:
    return [str(model.actuator(i).name) for i in range(int(model.nu))]


def _agent_index_from_name(name: str, num_agents: int) -> int | None:
    """Map Point-style actuator names to agent index (``x_0`` / ``x`` / ``x1``)."""
    if "_" in name:
        tail = name.rsplit("_", 1)[-1]
        if tail.isdigit():
            idx = int(tail)
            return idx if 0 <= idx < num_agents else None
    # Legacy multi_point.xml: ``x``, ``z`` → agent 0; ``x1``, ``z1`` → agent 1
    digits = "".join(ch for ch in name if ch.isdigit())
    if digits:
        idx = int(digits)
        return idx if 0 <= idx < num_agents else None
    return 0 if num_agents >= 1 else None


def classify_ctrl_layout(
    names: Sequence[str],
    *,
    num_agents: int,
    per_agent_dim: int,
) -> CtrlLayout:
    """Infer blocked vs interleaved from actuator name agent-index sequence."""
    if num_agents <= 1:
        return "blocked"

    ids: list[int] = []
    for name in names:
        idx = _agent_index_from_name(str(name), num_agents)
        if idx is None:
            raise ValueError(
                f"Cannot parse agent index from actuator name {name!r} "
                f"(names={list(names)})"
            )
        ids.append(idx)

    blocked = [i // per_agent_dim for i in range(num_agents * per_agent_dim)]
    interleaved = [i % num_agents for i in range(num_agents * per_agent_dim)]
    if ids == blocked:
        return "blocked"
    if ids == interleaved:
        return "interleaved"
    raise ValueError(
        f"Unrecognized MA ctrl layout for {num_agents} agents "
        f"(per_dim={per_agent_dim}): names={list(names)} agent_ids={ids}; "
        f"expected blocked={blocked} or interleaved={interleaved}"
    )


def pack_ma_actions_to_ctrl(
    action: dict[str, Any],
    agents: Sequence[str],
    model: Any,
    *,
    layout: CtrlLayout | None = None,
) -> np.ndarray:
    """Pack ``{agent_i: (per_dim,)}`` into length-``nu`` ctrl matching ``model`` actuators."""
    num_agents = len(agents)
    nu = int(model.nu)
    if nu % num_agents != 0:
        raise ValueError(f"model.nu={nu} not divisible by num_agents={num_agents}")
    per_agent_dim = nu // num_agents

    acts: list[np.ndarray] = []
    for agent in agents:
        vec = np.asarray(action[agent], dtype=np.float64).reshape(-1)
        if vec.shape != (per_agent_dim,):
            raise ValueError(
                f"Action dimension mismatch for {agent}: {vec.shape} vs {(per_agent_dim,)}"
            )
        acts.append(vec)

    names = actuator_names(model)
    resolved = layout or classify_ctrl_layout(
        names, num_agents=num_agents, per_agent_dim=per_agent_dim,
    )

    out = np.zeros(nu, dtype=np.float64)
    if resolved == "blocked":
        for index, vec in enumerate(acts):
            out[index * per_agent_dim : (index + 1) * per_agent_dim] = vec
    else:
        # Interleaved: [a0_d0, a1_d0, ..., a0_d1, a1_d1, ...]
        out[:] = np.stack(acts, axis=1).flatten()
    return out
