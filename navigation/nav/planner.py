"""Global path planner: A* search on a costmap grid."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .A_star import AStarPlanner

def global_plan(
    start: Tuple[float, float],
    goal: Tuple[float, float],
    costmap: np.ndarray,
) -> List[Tuple[float, float]]:
    planner = AStarPlanner(costmap)
    path = planner.plan(start, goal)
    return path
