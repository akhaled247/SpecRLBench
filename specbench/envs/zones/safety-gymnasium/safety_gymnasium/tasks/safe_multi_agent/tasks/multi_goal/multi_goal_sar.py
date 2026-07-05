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

import gymnasium
import mujoco

from safety_gymnasium.tasks.safe_multi_agent.bases.base_task import BaseTask
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms import LtlWalls
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms import Walls
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.zones import Zones
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.buildings import Buildings
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.casualtys import Casualtys
from safety_gymnasium.tasks.safe_multi_agent.assets.mocaps.gremlins import Gremlins
from safety_gymnasium.tasks.safe_multi_agent.utils.sar_utils import *

# Ring layout for interior walls
WALL_RING_RADIUS = 1.75
WALL_BASE_HALF_SIZES = [0.1, 0.3, 0.2]
WALL_COUNT = 20
BUILDING_KEEPOUT = 0.25
BUILDING_BORDER_SIDE_LENGTH = 4.5
BUILDING_MARGIN = 0.8
HUMAN_KEEPOUT = 0.2
MARGIN = 1.25

class MultiGoalSAR(BaseTask):
    """Multi-agent zone navigation with optional ring-placed interior walls."""
    _cached_wall_half_sizes = None
    _cached_building_locations = None

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.placements_conf.extents = [-3.5, -3.5, 3.5, 3.5]
        self.lidar_conf.num_bins = 16
        self.lidar_conf.max_dist = 2
        self.lidar_conf.exp_gain = 0.5
        self.lidar_conf.alias = True
        self.lidar_conf.type = 'natural' # choices: 'pseudo' 'natural'
        self.cost_conf.constrain_indicator = False
        self.observation_flatten = False

        self._add_geoms(
            LtlWalls(),
            Buildings(
                color=list(Buildings.COLORS)[0],
                size=BUILDING_KEEPOUT*0.75,
                num=self.agent_num,
                keepout=BUILDING_KEEPOUT,
                placements = border_placements(
                    BUILDING_BORDER_SIDE_LENGTH,
                    BUILDING_MARGIN)),
            Casualtys(
                category=list(Casualtys.CATEGORIES)[-2],
                size=0.05,
                num=self.agent_num-(self.agent_num//2),
                keepout=HUMAN_KEEPOUT,
            ),
            Casualtys(
                num=self.agent_num//2,
                category=list(Casualtys.CATEGORIES)[-1],
                size=0.05,),
            Walls(num=WALL_COUNT),
        )

        self._add_mocaps(
            Gremlins(num=config['agent_num'], size=0.28, dist_threshold=0.0, keepout=0.0)
        )

    def calculate_reward(self):
        return {f'agent_{i}': 0.0 for i in range(self.agent.agent_num)}

    def specific_reset(self):
        # print(f"BUILDING ATTR: {dir(self._geoms.get('tan_buildings'))})
        # print(f"BUILDING POS: {self._geoms.get('tan_buildings').pos}")
        pass

    def specific_step(self):
        pass

    def update_world(self):
        pass

    def _replace_geom(self, geom) -> None:
        """Update _geoms like _add_geoms but without duplicate registration checks."""
        self._geoms[geom.name] = geom
        setattr(self, geom.name, geom)
        geom.set_agent(self.agent)

    def _build(self):
        # randomized wall sizes that persist between runs
        self._cached_wall_half_sizes = size_randomization(
            WALL_BASE_HALF_SIZES, 
            WALL_COUNT, 
            margins=(np.array(WALL_BASE_HALF_SIZES)/2).tolist(), #make tolerance half of size 
            random_generator=self.random_generator
            ) if self._cached_wall_half_sizes is None else self._cached_wall_half_sizes
        self._replace_geom(Walls(
                num=WALL_COUNT,
                placements=ring_placements(
                    WALL_RING_RADIUS, WALL_COUNT, margin=MARGIN
                ),
                half_sizes=self._cached_wall_half_sizes,
                keepout=0.4,
            ))
        
        # randomized building/entrapped casualty locations that persist between runs
        self._cached_building_locations = border_locations(
            BUILDING_BORDER_SIDE_LENGTH,
            BUILDING_MARGIN,
            BUILDING_KEEPOUT,
            self.agent_num,
            random_generator=self.random_generator
            ) if self._cached_building_locations is None else self._cached_building_locations
        self._replace_geom(Buildings(
                color=list(Buildings.COLORS)[0],
                size=BUILDING_KEEPOUT*0.75,
                num=self.agent_num,
                keepout=0.0,
                locations = self._cached_building_locations,
                debug=True
                    ))
        self._replace_geom(Casualtys(
                category=list(Casualtys.CATEGORIES)[-1],
                size=0.05,
                num=self.agent_num//2,
                keepout=0.0,
                locations = self._cached_building_locations))
                
        return super()._build()

    @property
    def goal_achieved(self):
        return tuple(False for _ in range(self.agent.agent_num))
