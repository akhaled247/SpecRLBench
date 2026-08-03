"""Smoke / customize entrypoint for CustomizedSAR (multi-agent SAR).

Edit knobs in ``specbench.envs.zones.customized_sar_env_config``, then:

  python SpecRLBench/customize_sar_env.py

Eval (WC paper stack)::

  python eval_safepo_sa_on_ma_env.py --run-dir ... --eval-env CustomizedSARWC-v0
"""

from __future__ import annotations

import safety_gymnasium
import specbench  # noqa: F401  # registers SpecRLBench env helpers

from specbench.envs.zones.customized_sar_env_config import (
    CUSTOMIZED_SAR_ENV_CONFIG,
    ensure_customized_sar_registered,
)

# Back-compat alias for scripts that import env_config from this module.
env_config = CUSTOMIZED_SAR_ENV_CONFIG


def main() -> None:
    env_id = ensure_customized_sar_registered()
    env = safety_gymnasium.make(env_id, render_mode="human")

    obs, info = env.reset(seed=0)

    for _ in range(200):
        action = {
            agent: env.action_space(agent).sample()
            for agent in env.unwrapped.possible_agents
        }
        env.step(action)
        env.render()

    env.close()


if __name__ == "__main__":
    main()
