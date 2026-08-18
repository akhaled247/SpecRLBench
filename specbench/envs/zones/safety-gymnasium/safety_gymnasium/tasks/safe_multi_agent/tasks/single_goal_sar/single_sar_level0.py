# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Multi Goal with a SAR environment."""

from collections import OrderedDict

import gymnasium
import mujoco
import numpy as np

from safety_gymnasium.tasks.safe_multi_agent.bases.base_task import BaseTask
from safety_gymnasium.tasks.safe_multi_agent.world import World
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms import LtlWalls
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.buildings import Buildings
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.casualtys import Casualtys
from safety_gymnasium.tasks.safe_multi_agent.assets.mocaps.gremlins import Gremlins
from safety_gymnasium.tasks.safe_multi_agent.tasks.multi_goal_sar.sar_config_loader import (
    apply_sar_recipe,
)
from safety_gymnasium.tasks.safe_multi_agent.utils.sar_utils import (
    agent_inside_building_idx,
    border_placements,
    building_geom,
    clear_building_pinned_locations,
    clamp_building_placement_keepout,
    mission_goal_achieved,
    sync_building_dependents_into_layout,
)


class SingleGoalSARLevel0(BaseTask):
    """Multi-agent zone navigation with optional ring-placed interior walls."""

    # Level identity (stock L0). Shared recipe lives in configs/multi_goal_sar.yaml.
    wall_count = 0
    reward_goal = 1.0
    surface_casualties_per_agent: int = 1
    entrapped_casualties_per_agent: int = 0
    building_num: int = 0

    # Placeholders so BaseTask._parse can accept CustomizedSAR overrides.
    wall_ring_radius = 2.0
    wall_margin = 1.0
    wall_base_half_sizes = [0.1, 0.3, 0.2]
    walls_keepout = 0.4
    walls_half_size_randomization = True
    building_keepout = 0.4
    building_border_side_length = 4.5
    building_margin = 0.8
    casualty_size = 0.05
    casualty_touch_offset = 0.15
    casualty_keepout = 0.2
    entrapped_casualty_keepout = 0.0
    agent_keepout = 0.25
    agent_placements = [(-0.67, -0.67, 0.67, 0.67)]
    gremlin_size = 0.175
    gremlin_dist_threshold = 0.175
    gremlin_keepout = 0.0
    building_perimeter_wall_height = 0.75
    building_perimeter_wall_collision_threshold = 8.0

    def __init__(self, config) -> None:
        self._cached_wall_half_sizes = None
        self._cached_building_rots = None
        config = dict(config)
        skip_keys = frozenset(config.pop('_sar_skip_constant_keys', ()))
        super().__init__(config=config)

        apply_sar_recipe(self, skip_keys=skip_keys)
        # Stock levels keep fixed lidar resolution (customized_defaults is CustomizedSAR-only).
        if 'lidar_conf.num_bins' not in skip_keys and 'num_bins' not in skip_keys:
            self.lidar_conf.num_bins = 16

        self.last_dist_casualty = None
        self._buildings_entered: set[int] = set()
        self._lidar_suppressed_geom_ids: set[int] = set()
        # Lagged skip set: updated by snapshot_casualty_lidar_skip at step start.
        self._casualty_lidar_skip_rows: dict[str, frozenset[int]] = {}
        # Per-agent last LOS world XY for surface casualties (sticky when occluded).
        self._surface_last_seen: dict[int, dict[int, np.ndarray]] = {}
        self._surface_sticky_active: dict[int, set[int]] = {}

        # Spawn agents in a specified area
        self._build_agent(
            self.agent_name,
            keepout=self.agent_keepout,
            placements=self.agent_placements,
        )
        surface_casualtys_int = int(self.agent_num * self.surface_casualties_per_agent)
        # One surface casualty for solo training; otherwise one per agent.
        self.casualty_num = self.agent_num
        self._add_geoms(
            LtlWalls(contype=1),
        )

        if surface_casualtys_int > 0:
            self._add_geoms(
                Casualtys(
                    category="surface",
                    size=self.casualty_size,
                    num=int(surface_casualtys_int),
                    keepout=self.casualty_keepout,
                ),
            )

        # Gremlin count is always agent_num (not user-configurable).
        self._add_mocaps(
            Gremlins(
                num=int(self.agent_num),
                size=self.gremlin_size,
                dist_threshold=self.gremlin_dist_threshold,
                keepout=self.gremlin_keepout,
            )
        )

    def _dist_to_casualty(self, agent_idx: int) -> float:
        if not hasattr(self, 'surface_casualtys'):
            return 0.0
        casualty_pos = self.surface_casualtys.pos[0]
        return self.agent.dist_xy(agent_idx, casualty_pos)

    def _dist_to_casualtys(self, agent_idx: int) -> list[float]:
        all_casualtys_dist = []
        if hasattr(self, 'surface_casualtys'):
            casualty_poses = (self.surface_casualtys.pos[i] 
                              for i in range(int(self.agent_num * self.surface_casualties_per_agent)))
            all_casualtys_dist.extend([self.agent.dist_xy(agent_idx, pos) for pos in casualty_poses])
        if hasattr(self, 'entrapped_casualtys'):
            casualty_poses = (self.entrapped_casualtys.pos[i] 
                              for i in range(int(self.agent_num * self.entrapped_casualties_per_agent)))
            all_casualtys_dist.extend([self.agent.dist_xy(agent_idx, pos) for pos in casualty_poses])
        # print(all_casualtys_dist)
        return all_casualtys_dist

    def _casualtys_rescued(self) -> list[float]:
        all_casualtys_rescued = []
        if hasattr(self, 'surface_casualtys'):
            all_casualtys_rescued.extend(self.surface_casualtys.rescued)
        if hasattr(self, 'entrapped_casualtys'):
            all_casualtys_rescued.extend(self.entrapped_casualtys.rescued)
        # print(all_casualtys_rescued)
        return all_casualtys_rescued   

    def build_observation_space(self) -> gymnasium.spaces.Dict:
        super().build_observation_space()
        buildings = building_geom(self)
        obs_space_dict = OrderedDict(self.obs_info.obs_space_dict.spaces)
        if buildings is not None:
            obs_space_dict[f'{buildings.color_name}_buildings_visited'] = gymnasium.spaces.Box(
                0.0,
                1.0,
                (buildings.num,),
                dtype=np.float64,
            )
        # Arena boundary can cost ``walls`` even when wall_count=0 — keep walls_lidar keys.
        if self._arena_ltl_walls() is not None:
            for i in range(self.agent_num):
                name = f'walls_lidar_{i}'
                if name not in obs_space_dict:
                    obs_space_dict[name] = gymnasium.spaces.Box(
                        0.0,
                        1.0,
                        (self.lidar_conf.num_bins,),
                        dtype=np.float64,
                    )
        self.obs_info.obs_space_dict = gymnasium.spaces.Dict(obs_space_dict)
        if self.observation_flatten:
            self.observation_space = gymnasium.spaces.utils.flatten_space(
                self.obs_info.obs_space_dict,
            )
        else:
            self.observation_space = self.obs_info.obs_space_dict
        return self.observation_space

    def process_obs(self, obs: dict) -> dict:
        """Split global buildings-visited vector into per-agent scalars for deploy parity."""
        processed = super().process_obs(obs)
        buildings = building_geom(self)
        if buildings is None:
            return processed
        key = f'{buildings.color_name}_buildings_visited'
        if key not in obs:
            return processed
        visited = np.asarray(obs[key], dtype=np.float64).reshape(-1)
        # Always emit a scalar per agent (SA flatten expects the key). When
        # building_num < agent_num, share the global any-entered bit.
        any_entered = float(np.any(visited)) if visited.size else 0.0
        for i in range(self.agent_num):
            agent = f'agent_{i}'
            if agent not in processed:
                continue
            value = float(visited[i]) if i < visited.size else any_entered
            processed[agent][key] = np.array([value], dtype=np.float64)
        return processed

    def calculate_reward(self):
        """Distance delta toward visible casualty and touch bonus."""
        rewards = {}
        touch_threshold = 0.0
        if hasattr(self, 'surface_casualtys'):
            touch_threshold = self.surface_casualtys.size + self.casualty_touch_offset
        if hasattr(self, 'entrapped_casualtys'):
            touch_threshold = self.entrapped_casualtys.size + self.casualty_touch_offset

        for i in range(self.agent_num):
            a = f'agent_{i}'
            reward = 0

            # Distance-based reward shaping
            dists = self._dist_to_casualtys(i)
            if not dists:
                rewards[a] = 0
                self.last_dist_casualty[i] = 0.0
                continue
            min_dist = min(dists)
            min_casualty_rescued = self._casualtys_rescued()[dists.index(min_dist)]
            # if min_dist <= touch_threshold: print('uh oh') 
            # else: print(min_dist)
            if min_dist <= touch_threshold and not min_casualty_rescued:
                # print('casualty found')
                reward += (self.reward_goal
                           / (self.agent_num * self.surface_casualties_per_agent
                              + self.agent_num * self.entrapped_casualties_per_agent))
            self.last_dist_casualty[i] = min_dist

            rewards[a] = reward
        return rewards

    def specific_reset(self):
        """Reset SAR-specific episode state after layout resample."""
        if hasattr(self, 'surface_casualtys'):
            self.surface_casualtys.rescued = [False] * self.surface_casualtys.num
        if hasattr(self, 'entrapped_casualtys'):
            self.entrapped_casualtys.rescued = [False] * self.entrapped_casualtys.num
        self.last_dist_casualty = [self._dist_to_casualty(i) for i in range(self.agent_num)]
        self._buildings_entered = set()
        self._lidar_suppressed_geom_ids = set()
        self._surface_last_seen = {i: {} for i in range(self.agent_num)}
        self._surface_sticky_active = {i: set() for i in range(self.agent_num)}
        self.snapshot_casualty_lidar_skip()
        buildings = building_geom(self)
        if buildings is not None:
            buildings.prev_contact = [False] * buildings.num
        self._sync_entered_building_state()

    def specific_step(self):
        self._sync_entered_building_state()
        self._sync_rescued_casualty_state()

    def _casualty_geoms(self):
        geoms = []
        if hasattr(self, 'surface_casualtys'):
            geoms.append(self.surface_casualtys)
        if hasattr(self, 'entrapped_casualtys'):
            geoms.append(self.entrapped_casualtys)
        return geoms

    def snapshot_casualty_lidar_skip(self) -> None:
        """Freeze lidar-skip rows from current ``rescued`` (call at step start)."""
        skip: dict[str, frozenset[int]] = {}
        for geom in self._casualty_geoms():
            rescued = getattr(geom, 'rescued', None)
            if rescued is None:
                continue
            skip[geom.name] = frozenset(i for i, flag in enumerate(rescued) if flag)
        self._casualty_lidar_skip_rows = skip

    def _sync_rescued_casualty_state(self) -> None:
        """Hide rescued casualties: alpha=0 immediately; lidar suppress lagged one step."""
        if not hasattr(self, 'model') or self.model is None:
            return
        suppressed = set(getattr(self, '_lidar_suppressed_geom_ids', set()))
        skip_map = getattr(self, '_casualty_lidar_skip_rows', {})
        for geom in self._casualty_geoms():
            rescued = getattr(geom, 'rescued', None)
            if rescued is None:
                continue
            lidar_skip = skip_map.get(geom.name, frozenset())
            for row, is_rescued in enumerate(rescued):
                geom_id = self._obstacle_geom_id_for_instance(geom, row)
                if geom_id is None:
                    continue
                if is_rescued:
                    self.model.geom_rgba[geom_id][-1] = 0.0
                    # Keep geom lidar-hit until skip snapshot catches up (rescue-step peak).
                    if row in lidar_skip:
                        suppressed.add(geom_id)
                    else:
                        suppressed.discard(geom_id)
                else:
                    suppressed.discard(geom_id)
                    self.model.geom_rgba[geom_id][-1] = float(geom.alpha)
        self._lidar_suppressed_geom_ids = suppressed

    def _rescued_casualty_rows(self, obstacle) -> frozenset[int]:
        """Rows omitted from casualty lidar (lagged snapshot when present)."""
        skip_map = getattr(self, '_casualty_lidar_skip_rows', None)
        if skip_map is not None:
            return skip_map.get(obstacle.name, frozenset())
        rescued = getattr(obstacle, 'rescued', None)
        if rescued is None:
            return frozenset()
        return frozenset(i for i, flag in enumerate(rescued) if flag)

    def _apply_surface_last_seen_lidar(self, obs: dict) -> None:
        """Update sticky last-seen XY; inject synthetic surface lidar when occluded."""
        surface = getattr(self, 'surface_casualtys', None)
        if surface is None or not getattr(surface, 'is_lidar_observed', False):
            return
        if not hasattr(self, '_surface_last_seen') or not self._surface_last_seen:
            self._surface_last_seen = {i: {} for i in range(self.agent_num)}
        if not hasattr(self, '_surface_sticky_active'):
            self._surface_sticky_active = {i: set() for i in range(self.agent_num)}

        skip = self._rescued_casualty_rows(surface)

        for agent_idx in range(self.agent_num):
            key = f'{surface.name}_lidar_{agent_idx}'
            if key not in obs:
                continue
            vals = np.asarray(obs[key], dtype=np.float64).copy()
            agent_seen = self._surface_last_seen.setdefault(agent_idx, {})
            sticky_rows = self._surface_sticky_active.setdefault(agent_idx, set())
            sticky_rows.clear()

            for row in list(agent_seen.keys()):
                if row in skip:
                    del agent_seen[row]

            for row in range(int(surface.num)):
                if row in skip:
                    continue
                pos = np.asarray(self._lidar_target_pos(agent_idx, surface, row), dtype=float)
                los = bool(self._lidar_line_of_sight(agent_idx, pos, surface, row))
                if los:
                    agent_seen[row] = np.asarray(pos[:2], dtype=float).copy()
                elif row in agent_seen:
                    self._accumulate_pseudo_lidar_reading(
                        vals, agent_idx, agent_seen[row],
                    )
                    sticky_rows.add(row)
            obs[key] = vals

    def _sync_entered_building_state(self) -> None:
        """Sticky-hide entered building shells for the rest of the episode."""
        buildings = building_geom(self)
        if buildings is None or not hasattr(self, 'model') or self.model is None:
            # Still refresh casualty hide when no buildings
            self._sync_rescued_casualty_state()
            return

        for agent_idx in range(self.agent_num):
            inside_idx = agent_inside_building_idx(self, agent_idx)
            if inside_idx is not None:
                self._buildings_entered.add(inside_idx)

        suppressed: set[int] = set()
        for row in self._buildings_entered:
            geom_id = self._obstacle_geom_id_for_instance(buildings, row)
            if geom_id is not None:
                suppressed.add(geom_id)

        self._lidar_suppressed_geom_ids = suppressed

        for row in range(buildings.num):
            geom_id = self._obstacle_geom_id_for_instance(buildings, row)
            if geom_id is None:
                continue
            if row in self._buildings_entered:
                self.model.geom_rgba[geom_id][-1] = 0.0
            else:
                self.model.geom_rgba[geom_id][-1] = buildings.alpha

        # Merge rescued casualty suppression after buildings
        self._sync_rescued_casualty_state()

    def update_world(self):
        pass

    def _prepare_layout(self) -> None:
        buildings = building_geom(self)
        has_buildings = buildings is not None and buildings.num > 0
        if has_buildings:
            clear_building_pinned_locations(self)
            clamp_building_placement_keepout(self, self.building_margin)
        if has_buildings or self.placements_conf.placements is None:
            self._build_placements_dict()
            self.random_generator.set_placements_info(
                self.placements_conf.placements,
                self.placements_conf.extents,
                self.placements_conf.margin,
            )
        if self.random_generator.agent_num is None:
            self.random_generator.agent_num = self.agent.agent_num
        self.world_info.layout = self.random_generator.build_layout()
        if has_buildings:
            sync_building_dependents_into_layout(self, self.world_info.layout)

    def _fast_resample_layout(self) -> None:
        self._prepare_layout()
        self.world_info.world_config_dict = self._build_world_config(self.world_info.layout)
        self._apply_layout_from_config()

    def _build(self):
        self._prepare_layout()
        self.world_info.world_config_dict = self._build_world_config(self.world_info.layout)
        if self.world is None:
            self.world = World(self.agent, self._obstacles, self.world_info.world_config_dict)
            self.world.reset()
            self.world.build()
        else:
            self.world.reset(build=False)
            self.world.rebuild(self.world_info.world_config_dict, state=False)
            if self.viewer:
                self._update_viewer(self.model, self.data)

    def _replace_geom(self, geom) -> None:
        """Update _geoms like _add_geoms but without duplicate registration checks."""
        self._geoms[geom.name] = geom
        setattr(self, geom.name, geom)
        geom.set_agent(self.agent)

    def _replace_border_buildings(self, num=None) -> None:
        self._replace_geom(Buildings(
            color=list(Buildings.COLORS)[0],
            size=self.building_keepout * 0.75,
            num=int(self.agent_num if num is None else num),
            keepout=self.building_keepout,
            placements=border_placements(
                self.building_border_side_length,
                self.building_margin,
            ),
        ))

    def _replace_building_perimeter_walls(self) -> None:
        if self.building_num <= 0:
            return
        factor = self.building_keepout * 0.75
        for i in range(self.building_num):
            self._replace_geom(LtlWalls(
                name=f'building{i}_ltl_walls',
                locate_factor=factor,
                size=factor,
                height=self.building_perimeter_wall_height,
                collision_threshold=self.building_perimeter_wall_collision_threshold,
            ))

    def _arena_ltl_walls(self):
        """Arena boundary ``LtlWalls`` (name ``ltl_walls``), not building perimeters."""
        arena = getattr(self, 'ltl_walls', None)
        if arena is not None and getattr(arena, 'name', None) == 'ltl_walls':
            return arena
        for obstacle in self._obstacles:
            if getattr(obstacle, 'name', None) == 'ltl_walls':
                return obstacle
        return None

    def _arena_walls_lidar(self, agent_idx: int) -> np.ndarray:
        """Pseudo lidar to arena boundary surfaces (no LOS; arena not lidar-observable)."""
        arena = self._arena_ltl_walls()
        bins = int(self.lidar_conf.num_bins)
        if arena is None or not getattr(arena, 'num', 0):
            return np.zeros(bins, dtype=np.float64)
        positions = [
            self._lidar_target_pos(agent_idx, arena, row)
            for row in range(int(arena.num))
        ]
        return self._obs_lidar_pseudo_new(agent_idx, positions)

    def _merge_arena_into_walls_lidar(self, obs: dict) -> None:
        """Fold arena ``ltl_walls`` into ``walls_lidar_*`` (max-pool with interior)."""
        if self._arena_ltl_walls() is None:
            return
        for i in range(self.agent_num):
            key = f'walls_lidar_{i}'
            arena_lidar = self._arena_walls_lidar(i)
            if key in obs:
                obs[key] = np.maximum(np.asarray(obs[key], dtype=np.float64), arena_lidar)
            else:
                obs[key] = arena_lidar

    def try_lidar_ids(self, obstacle, obs, i, skip_instance_rows=None):
        """pseudo_occluded lidar with per-instance line-of-sight (walls block view)."""
        skip_rows = skip_instance_rows or frozenset()
        is_occluded = False
        if (
            hasattr(obstacle, 'is_lidar_ids_observed')
            and obstacle.is_lidar_ids_observed
            and self.lidar_conf.type == 'pseudo_occluded'
        ):
            lidar, lidar_ids = self._obs_lidar_pseudo_occluded_new(
                i, obstacle, return_ids=True, skip_instance_rows=skip_rows,
            )
            obs[f"{obstacle.name}_lidar_{i}"] = lidar
            obs[f"{obstacle.name}_lidar_ids_{i}"] = lidar_ids
        elif not is_occluded:
            positions = [
                self._lidar_target_pos(i, obstacle, row)
                for row in range(obstacle.num)
                if row not in skip_rows
            ]
            obs[f"{obstacle.name}_lidar_{i}"] = self._obs_lidar_pseudo_new(i, positions)
        else:
            obs[f"{obstacle.name}_lidar_{i}"] = self._obs_lidar_pseudo_occluded_new(
                i, obstacle, skip_instance_rows=skip_rows,
            )

    def obs(self) -> dict | np.ndarray:
        """Return the observation of our agent."""
        # pylint: disable-next=no-member
        mujoco.mj_forward(self.model, self.data)  # Needed to get sensor's data correct
        self._sync_entered_building_state()
        obs = {}

        obs.update(self.agent.obs_sensor())

        # observations of obstacles
        for obstacle in self._obstacles:
            if obstacle.is_lidar_observed:
                if 'gremlins' in obstacle.name:
                    for i in range(self.agent_num):
                        name = f"{obstacle.name}_lidar_{i}"
                        poses = obstacle.pos.copy()
                        del poses[i]
                        obs[name] = self._obs_lidar_new(
                            i, poses, obstacle.group, obstacle=obstacle,
                        )
                elif obstacle.name.endswith('_buildings'):
                    skip_rows = frozenset(self._buildings_entered)
                    for i in range(self.agent_num):
                        self.try_lidar_ids(obstacle, obs, i, skip_instance_rows=skip_rows)
                elif 'casualtys' in obstacle.name:
                    skip_rows = self._rescued_casualty_rows(obstacle)
                    for i in range(self.agent_num):
                        self.try_lidar_ids(obstacle, obs, i, skip_instance_rows=skip_rows)
                else:
                    for i in range(self.agent_num):
                        self.try_lidar_ids(obstacle, obs, i)

            if hasattr(obstacle, 'is_comp_observed') and obstacle.is_comp_observed:
                obs[obstacle.name + '_comp'] = self._obs_compass(obstacle.pos)

        self._apply_surface_last_seen_lidar(obs)
        self._merge_arena_into_walls_lidar(obs)

        buildings = building_geom(self)
        if buildings is not None:
            visited = np.zeros(buildings.num, dtype=np.float64)
            for row in self._buildings_entered:
                visited[row] = 1.0
            obs[f'{buildings.color_name}_buildings_visited'] = visited

        if self.observe_vision:
            for i in range(self.agent_num):
                name = f'vision_{i}'
                obs[name] = self._obs_vision(camera_name=name)
        if self.observation_flatten:
            obs = gymnasium.spaces.utils.flatten(self.obs_info.obs_space_dict, obs)
        return obs

    @property
    def goal_achieved(self):
        return mission_goal_achieved(self)
