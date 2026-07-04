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
"""Multi Goal level 3."""

from safety_gymnasium.tasks.safe_multi_agent.bases.base_task import BaseTask
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms import LtlWalls
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms import Walls
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.zones import Zones
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.buildings import Buildings
from safety_gymnasium.tasks.safe_multi_agent.assets.mocaps.gremlins import Gremlins
from safety_gymnasium.tasks.safe_multi_agent.utils.wall_utils import *

# Ring layout for interior walls
WALL_RING_RADIUS = 1.75
WALL_COUNT = 20
MARGIN = 1.25

class MultiGoalLevel3(BaseTask):
    """Multi-agent zone navigation with optional ring-placed interior walls."""
    _cached_wall_half_sizes = None
    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.placements_conf.extents = [-3.5, -3.5, 3.5, 3.5]
        self.lidar_conf.num_bins = 16
        self.lidar_conf.max_dist = 2
        self.lidar_conf.exp_gain = 0.5
        self.lidar_conf.alias = True
        self.lidar_conf.type = str('natural') # choices: 'pseudo' 'natural'
        self.cost_conf.constrain_indicator = False
        self.observation_flatten = False

        self._add_geoms(LtlWalls())

        zone_size = 0.5
        keepout=0.5
        self._add_geoms(
            Buildings(color='black', 
                  size=0.3,
                  num=6,
                  keepout=keepout,
                  placements = border_placements(side_length=4, thickness=keepout*4)),
            Zones(color='magenta', size=zone_size / 3, num=3, keepout=zone_size / 3 + 0.15),
            Zones(color='red', size=zone_size / 2, num=2, keepout=zone_size / 2 * 1.25),
            Walls(
                num=WALL_COUNT,
                placements=ring_placements(
                    WALL_RING_RADIUS, WALL_COUNT, margin=MARGIN
                ),
                half_sizes=[0.1, 0.3, 0.2],
                keepout=0.4,
            ),
        )

        self._add_mocaps(
            Gremlins(num=config['agent_num'], size=0.28, dist_threshold=0.0, keepout=0.0)
        )

    def calculate_reward(self):
        return {f'agent_{i}': 0.0 for i in range(self.agent.agent_num)}

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def update_world(self):
        pass
    
    def _build(self):
        self._cached_wall_half_sizes = size_randomization([0.1, 0.3, 0.2], WALL_COUNT, margins=[0.05, 0.15, 0.1], random_generator=self.random_generator) if self._cached_wall_half_sizes is None else self._cached_wall_half_sizes
        self._geoms.update({Walls.name: Walls(
                num=WALL_COUNT,
                placements=ring_placements(
                    WALL_RING_RADIUS, WALL_COUNT, margin=MARGIN
                ),
                half_sizes=self._cached_wall_half_sizes,
                keepout=0.4,
            )})
        return super()._build()

    @property
    def goal_achieved(self):
        return tuple(False for _ in range(self.agent.agent_num))
