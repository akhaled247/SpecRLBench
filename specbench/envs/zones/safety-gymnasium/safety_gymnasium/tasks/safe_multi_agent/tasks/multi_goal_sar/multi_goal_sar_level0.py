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
"""Multi Goal SAR level 3: walls + buildings with surface and entrapped casualties."""

from safety_gymnasium.tasks.safe_multi_agent.tasks.single_goal_sar.single_sar_level2 import (
    SingleGoalSARLevel2,
)


class MultiGoalSARLevel0(SingleGoalSARLevel2):
    """L2 setup with both surface and entrapped casualties enabled."""

    wall_count = 0
    surface_casualties_per_agent = 1
    entrapped_casualties_per_agent = 1

    def all_entrapped_casualtys_rescued(self) -> bool:
        if hasattr(self, 'entrapped_casualtys'):
            return all(self.entrapped_casualtys.rescued)
        else:
             return False
        
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
            min_casualty_is_entrapped = dists.index(min_dist) >= self.surface_casualtys.num
            
            # if min_dist <= touch_threshold: print('uh oh') 
            # else: print(min_dist)
            if (min_dist <= touch_threshold 
                and not min_casualty_rescued
                and (min_casualty_is_entrapped
                     or (not min_casualty_is_entrapped and self.all_entrapped_casualtys_rescued()))
                     ):
                # print('casualty found')
                reward += (self.reward_goal
                            / (self.agent_num * self.surface_casualties_per_agent
                                + self.agent_num * self.entrapped_casualties_per_agent))
            self.last_dist_casualty[i] = min_dist

            rewards[a] = reward
        return rewards
