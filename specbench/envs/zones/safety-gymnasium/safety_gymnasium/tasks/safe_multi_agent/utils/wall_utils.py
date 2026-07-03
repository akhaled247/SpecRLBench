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
"""Utils for wall placement on rings and arcs."""

from __future__ import annotations

import numpy as np
from safety_gymnasium.tasks.safe_multi_agent.utils.random_generator import RandomGenerator


def ring_locations(radius: float, n: int) -> list[tuple[float, float]]:
    """Fixed (x, y) centers evenly spaced on a circle."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(float(radius * np.cos(theta)), float(radius * np.sin(theta))) for theta in angles]


def ring_placements(
    radius: float,
    n: int,
    margin: float | None = None,
    keepout: float = 0.0,
) -> list[tuple[float, float, float, float]]:
    """Axis-aligned sampling boxes centered on ring points (for random placement).

    Sampler shrinks each box by ``keepout`` on all sides; need ``margin > keepout``
  or ``draw_placement`` asserts with no valid rectangles.
    """
    if margin is None:
        margin = keepout + 0.05
    margin = max(margin, keepout + 1e-3)
    boxes = []
    for x, y in ring_locations(radius, n):
        boxes.append((x - margin, y - margin, x + margin, y + margin))
    return boxes

def size_randomization(
    base_half_sizes: list,
    n: int,
    margin: float | None = None,
    margins: list[float] = [0, 0, 0],
    random_generator: RandomGenerator | None = None,
) -> list[list[float, float, float]]:
    """Axis-aligned sampling boxes centered on ring points (for random placement).
    Sampler shrinks each box by ``keepout`` on all sides; need ``margin > keepout``
  or ``draw_placement`` asserts with no valid rectangles.
    """
    if margin != None: margins = [margin, margin, margin] if margins == [0, 0, 0] else margins
    x, y, z = base_half_sizes
    x_margin, y_margin, z_margin = margins

    assert any(np.array(base_half_sizes)-np.array(margins) > 0), "Margins should ensure non-negative values" 
    
    return np.array([random_generator.uniform(
        x-x_margin, x+x_margin, n),
        random_generator.uniform(
        y-y_margin, y+y_margin, n),
        random_generator.uniform(
        z-z_margin, z+z_margin, n)]).transpose()
