from specbench.envs.zones.safety_gym_wrapper_sar_wc import SafetyGymWrapperMASARWC
from safety_gymnasium.tasks.safe_multi_agent.utils.sar_utils import (
    all_entrapped_casualties_rescued,
)


class SafetyGymWrapperMASARLTL(SafetyGymWrapperMASARWC):
    """Wall-cost (WC) SAR wrapper: terminate + ``info['cost']`` on ``cost_walls``.

    Parent ``SafetyGymWrapperMASAR`` with ``flat=True`` already collapses
    ``terminated`` to a bool. Must set that bool (not ``terminated[agent]``).
    """

    _cost_keys = ["cost_walls", "cost_collision"]
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["cost"] = 0
        hit = False
        for a in self.env.unwrapped.possible_agents:
            agent_info = info.get(a) if isinstance(info, dict) else None
            if not isinstance(agent_info, dict):
                continue
            for _cost_key in self._cost_keys:
                if float(agent_info.get(_cost_key, 0) or 0) > 0:
                    info["cost"] += 1
                    hit = True
                    if isinstance(terminated, dict):
                        terminated[a] = True
            if (float(agent_info.get('cost_casualtys_surface', 0) or 0) > 0
                and not all_entrapped_casualties_rescued(self.env.unwrapped.task)):
                # print('fajfl;kds;ljdsafa')
                info["cost"] += 1
                hit = True
                if isinstance(terminated, dict):
                    terminated[a] = True
        if hit and not isinstance(terminated, dict):
            terminated = True
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info["cost"] = 0
        return obs, info
