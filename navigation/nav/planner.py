"""Global path planner: A* search on a costmap grid."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from nav.A_star import AStarPlanner

def global_plan(
    start: Tuple[float, float],
    goal: Tuple[float, float],
    costmap: np.ndarray,
) -> List[Tuple[float, float]]:
    """
    Run path search over `costmap` to find a path from `start` to `goal`.

    Parameters
    ----------
    start : Tuple[float, float], (x, y)
        Start position in world (grid-unit) coordinates. `costmap[int(y), int(x)]`
        is the cell containing this point.
    goal : Tuple[float, float], (x, y)
        Goal position in the same coordinate system.
    costmap : np.ndarray, shape (rows, cols), dtype uint8
        Per-cell traversal cost. Cells with large cost are treated as impassable
        (lethal). Otherwise the cell's cost is added to the step cost so the
        planner is biased away from inflated/dangerous areas.

    Returns
    -------
    path : List[Tuple[float, float]], list of (x, y) waypoints from start to goal.
        World-coordinate waypoints from start to goal, inclusive of both ends.
        Returns [] if no path exists or if start/goal lie inside a lethal cell.

    Notes
    -----
    - Use 8-connectivity (N/S/E/W + 4 diagonals). Step cost between adjacent
      cells should be `dist + cell_cost`, where `dist` is 1.0 for cardinal moves
      and sqrt(2) for diagonals.
    - Use either a shortest path algorithm (like Dijkstra) or a heuristic search
      algorithm (like A*). If using A*, a good heuristic is the octile distance
      (diagonal distance) or Euclidean distance.
    """
    # TODO: Implement path search on the costmap grid to find a path from start to goal.
    planner = AStarPlanner(costmap)
    path = planner.plan(start, goal)
    return path
