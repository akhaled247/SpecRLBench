"""Shared CustomizedSAR knobs for smoke script + eval factory.

Edit here (or SpecRLBench/customize_sar_env.py re-exports). Eval loads
``CustomizedSAR-v0`` / ``CustomizedSARWC-v0`` via ``make_zone_env``.
"""

from __future__ import annotations

CUSTOMIZED_SAR_ENV_CONFIG: dict = {
    "env_id": "CustomizedSAR-v0",
    "agent_name": "Point",
    "max_episode_steps": 1000,
    "agent_num": 2,
    "building_num": 1,  # 0 → no buildings; omit/None → agent_num * entrapped
    "wall_count": 0,
    "surface_casualties_per_agent": 0.5,
    "entrapped_casualties_per_agent": 0.5,
    "reward_goal": 1.0,
    "lidar_conf.num_bins": 16,
}


def ensure_customized_sar_registered() -> str:
    """Register ``CustomizedSAR-v0`` once; return base env id (no WC/AC)."""
    from safety_gymnasium.utils.registration import safe_registry

    from specbench.envs.zones.safety_gym_register import register_helper

    env_id = str(CUSTOMIZED_SAR_ENV_CONFIG["env_id"])
    if env_id not in safe_registry:
        register_helper(env_config=dict(CUSTOMIZED_SAR_ENV_CONFIG))
    return env_id
