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
"""Wall."""

from dataclasses import dataclass, field

import numpy as np

from safety_gymnasium.tasks.safe_multi_agent.assets.color import COLOR
from safety_gymnasium.tasks.safe_multi_agent.assets.group import GROUP
from safety_gymnasium.tasks.safe_multi_agent.bases.base_object import Geom


@dataclass
class Walls(Geom):  # pylint: disable=too-many-instance-attributes
    """Walls - barriers in the environment not associated with any constraint.

    # NOTE: this is probably best to be auto-generated than manually specified.
    """

    name: str = 'walls'
    num: int = 0  # Number of walls
    half_size_x: float = 0.4  # Size of walls
    half_size_y: float = 0.2
    half_size_z: float = 0.2  # Height of walls
    placements: list = None  # Walls placements list (defaults to full extents)
    locations: list = field(default_factory=list)  # Fixed locations to override placements
    cost: float = 1.0  # Cost (per step) for being in contact with a wall

    keepout: float = 0.5  # This should not be used

    color: np.array = COLOR['wall']
    group: np.array = GROUP['wall']
    is_lidar_observed: bool = True
    is_constrained: bool = False

    # pylint: disable-next=too-many-arguments
    def get_config(self, xy_pos, rot):
        """To facilitate get specific config for this object."""
        return {
            'name': self.name,
            'size': [self.half_size_x, self.half_size_y, self.half_size_z],
            'pos': np.r_[xy_pos, self.half_size_z],
            'rot': rot,
            'type': 'box',
            'group': self.group,
            'rgba': self.color,
        }
    
    def cal_cost(self):
        """Contacts processing."""
        cost = {}
        if not self.is_constrained:
            return cost
        cost['cost_walls'] = 0
        for contact in self.engine.data.contact[: self.engine.data.ncon]:
            geom_ids = [contact.geom1, contact.geom2]
            geom_names = sorted([self.engine.model.geom(g).name for g in geom_ids])
            if any(n.startswith('wall') for n in geom_names) and any(
                n in self.agent.body_info[0].geom_names for n in geom_names
            ):
                # pylint: disable-next=no-member
                cost['cost_walls'] += self.cost

    @property
    def pos(self):
        """Helper to get list of wall positions."""
        # pylint: disable-next=no-member
        return [self.engine.data.body(f'wall{i}').xpos.copy() for i in range(self.num)]
