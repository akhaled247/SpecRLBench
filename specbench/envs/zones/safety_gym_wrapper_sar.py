from typing import Any

from specbench.utils.ltl.logic.assignment import Assignment
import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, WrapperObsType
from gymnasium.spaces import Box

from specbench.envs.zones.safety_gym_wrapper_ma import SafetyGymWrapperMA
from specbench.envs.zones.sar_propositions import (
    TEAM_PROPS,
    normalize_active_propositions,
    normalize_team_props,
    pick_per_agent_prop,
    should_expose_team_props,
    team_active_props,
)
from safety_gymnasium.tasks.safe_multi_agent.utils.sar_utils import (
    agent_inside_building_idx,
)


class SafetyGymWrapperMASAR(SafetyGymWrapperMA):
    """SAR wrapper: extends MA setup; overrides step/reset for rescue propositions."""

    flat = False
    action_dim = 2

    def __init__(self, env: Any, wall_sensor=True, flat=False):
        super().__init__(env, wall_sensor=wall_sensor)
        self.flat = flat
        self.prev_casualty_visible = False
        self.prev_entered_building = False

        # SAR proposition vocabulary (differs from LTL MA zone props).
        self.atomic_propositions = set()
        obs_space = env.observation_space
        if callable(obs_space):
            obs_space = obs_space(None)
        obs_keys = obs_space.spaces.keys()
        self.categories = set()
        for key in obs_keys:
            if "casualtys" in key.split('_'):
                # print(f"<safety_gym_wrapper_sar> {key.split('_')}")
                category = key.split('_')[0]
                self.categories.add(category)
                for i in range(self.num_agents):
                    self.atomic_propositions.add(category + '_' + str(i))

        task = env.unwrapped.task
        if should_expose_team_props(task):
            self.atomic_propositions.update(TEAM_PROPS)

        if self.flat:
            act_space = env.action_space
            if callable(act_space):
                act_space = Box(low=-1.0, high=1.0, shape=(self.num_agents * self.action_dim,))
            if isinstance(act_space, spaces.Box):
                self.action_space = act_space
            else:
                raise TypeError(f"Expected Box action space for SB3, got {type(act_space)}")

    def step(self, action: ActType):
        if self.flat:
            action = self.dictify_action(action)
        obs, reward, cost, terminated, truncated, info = gymnasium.Wrapper.step(self, action)

        if 'wall_sensor' in info["agent_0"]:
            for i, agent in enumerate(self.env.unwrapped.possible_agents):
                obs[agent][f'wall_sensor_{i}'] = info[agent]['wall_sensor']

        self.env.unwrapped.task.original_obs = obs

        info['propositions'] = []
        info['casualty_visible'] = False
        task = self.env.unwrapped.task
        expose_team = should_expose_team_props(task)
        for i, a in enumerate(self.env.unwrapped.possible_agents):
            agent_info: dict = info[a]
            prop = pick_per_agent_prop(self.categories, i, agent_info)
            if prop is not None:
                info['propositions'].append(prop)
            inside = agent_inside_building_idx(task, i) is not None
            entered = bool(getattr(task, '_buildings_entered', set()))
            if (
                f'entrapped_casualtys_lidar_{i}' in obs[a]
                and not (inside or entered)
            ):
                obs[a][f'entrapped_casualtys_lidar_{i}'] = np.zeros(
                    obs[a][f'entrapped_casualtys_lidar_{i}'].size,
                )

            if (
                f'surface_casualtys_lidar_{i}' in obs[a]
                and max(obs[a][f'surface_casualtys_lidar_{i}']) != 0.0
                and not self.prev_casualty_visible
            ):
                info['casualty_visible'] = True
                self.prev_casualty_visible = True

        if expose_team:
            info['propositions'].extend(normalize_team_props(team_active_props(task)))

        info['propositions'] = normalize_active_propositions(
            info['propositions'],
            num_agents=self.num_agents,
            categories=self.categories,
            include_team_props=expose_team,
        )

        mission_complete = all(self.env.unwrapped.task.goal_achieved)
        if not mission_complete and isinstance(info, dict):
            # Builder may stamp goal_met on agents / top-level before wrapper sees task flags.
            mission_complete = bool(info.get('goal_met')) or any(
                isinstance(info.get(a), dict) and info[a].get('goal_met')
                for a in getattr(self.env.unwrapped, 'possible_agents', [])
            )
        if mission_complete:
            info['goal_met'] = True

        if self.flat:
            obs = self.flatten_obs(obs)
            reward = float(np.mean(list(reward.values())))
            truncated = any(list(truncated.values()))
            terminated = any(list(terminated.values())) or mission_complete
        elif mission_complete:
            terminated = {a: True for a in self.env.unwrapped.possible_agents}

        return obs, reward, terminated, truncated, info

    def reset(
          self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        if seed is not None:
            self._layout_seed = seed
        elif hasattr(self, "_layout_seed"):
            self._layout_seed = (self._layout_seed + 1) % 100
            seed = self._layout_seed
        obs, info = gymnasium.Wrapper.reset(self, seed=seed, options=options)
        info['propositions'] = []
        info['casualty_visible'] = False
        self.prev_casualty_visible = False
        self.prev_entered_building = False
        for i, a in enumerate(self.env.unwrapped.possible_agents):
            obs[a][f'wall_sensor_{i}'] = np.array([0, 0, 0, 0])
        self.env.unwrapped.task.original_obs = obs
        if self.flat:
            obs = self.flatten_obs(obs)
        return obs, info

    def flatten_obs(self, obs):
        return {
            k: v
            for agent_obs in obs.values()
            for k, v in agent_obs.items()
        }

    def dictify_action(self, action) -> dict:
        return {
            f"agent_{i}": action[i * self.action_dim:(i + 1) * self.action_dim]
            for i in range(self.num_agents)
        }

    def get_possible_assignments(self) -> list[Assignment]:
            assignments = []
            agent_props: dict[int, set[str]] = {}
            team_props: set[str] = set()
            for prop in self.atomic_propositions:
                if prop in TEAM_PROPS:
                    team_props.add(prop)
                    continue
                agent_idx = int(prop.rsplit('_', 1)[1])
                agent_props.setdefault(agent_idx, set()).add(prop)
            per_agent_assignments = []
            for agent_id in range(self.num_agents):
                props = agent_props.get(agent_id, set())
                per_agent_assignments.append(Assignment.zero_or_one_propositions(props))
            team_choices = (
                Assignment.zero_or_one_propositions(team_props)
                if team_props else [Assignment()]
            )
            import itertools
            for combo in itertools.product(*per_agent_assignments, team_choices):
                merged = Assignment()
                for a in combo:
                    merged.update(a)
                assignments.append(merged)
            per_agent_expected = (len(self.categories) + 1) ** self.num_agents
            team_expected = (len(team_props) + 1) if team_props else 1
            expected = per_agent_expected * team_expected
            assert len(assignments) == expected, \
                f"Expected {expected} assignments, got {len(assignments)}"
            return assignments

    def get_propositions(self) -> list[str]:
        props = sorted(self.atomic_propositions)
        task = self.env.unwrapped.task
        if should_expose_team_props(task):
            props = sorted(set(props) | set(TEAM_PROPS))
        return props
