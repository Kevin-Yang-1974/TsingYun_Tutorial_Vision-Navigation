"""Local planner: Pure Pursuit controller."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def local_plan(
    current_pose: Tuple[float, float],
    max_speed: float,
    max_accel: float,
    global_path: List[Tuple[float, float]],
    costmap: np.ndarray = None,
) -> Tuple[float, float]:
    """
    Convert the next chunk of the global path into a velocity command.

    Parameters
    ----------
    current_pose : Tuple[float, float], (x, y)
        Current robot position in world (grid-unit) coordinates.
    max_speed : float
        Maximum allowed speed magnitude (grid units / second). The returned
        command vector should not exceed this length.
    max_accel : float
        Maximum allowed acceleration. You may ignore this if the world's
        `step()` already enforces a ramp; otherwise use it to compute a
        feasible command from the current velocity.
    global_path : List[Tuple[float, float]], list of (x, y) waypoints from start to goal
        Waypoints from the global planner, ordered from current pose to goal.
        May be empty if no path was found — in that case return `(0.0, 0.0)`.

    Returns
    -------
    cmd_vx, cmd_vy : float, float
        Desired world-frame velocity in grid units per second. The world step
        will clip this to `max_speed` and ramp toward it at `max_accel`, so
        returning a "pointing at the look-ahead" vector scaled to `max_speed`
        is usually the right move.

    Notes
    -----
    - Pure Pursuit recipe:
        1. Find the look-ahead point on `global_path`: walk forward from the
           closest waypoint to `current_pose` until the cumulative distance
           exceeds a look-ahead radius `Ld` (a tuning constant, e.g. 1.5-2.5
           grid units). If you reach the last waypoint first, use it.
        2. The command direction is `(look_ahead - current_pose)`, normalized.
        3. The command speed is `max_speed` (or a slowed value if the
           remaining path length is short, to ease into the goal).
    - Optional: More complex local programming methods (such as Dynamic
      Window Approach) can be used, or more complex model prediction methods
      (such as MPPI) can be tried.
    """
    Ld=3.5  # look-ahead distance in grid units
    slow_radius=6.0
    if not global_path:
        return 0.0, 0.0
    look_ahead_point = global_path[-1]  # default to the last waypoint
    for i, waypoint in enumerate(global_path):
        dist = np.sqrt((waypoint[0] - current_pose[0]) ** 2 + (waypoint[1] - current_pose[1]) ** 2)
        if dist >= Ld:
            look_ahead_point = waypoint
            break
    goal=global_path[-1]
    dist_to_goal = np.sqrt((goal[0] - current_pose[0]) ** 2 + (goal[1] - current_pose[1]) ** 2)
    if dist_to_goal<0.8:
        return 0.0, 0.0
    direction = np.array(look_ahead_point) - np.array(current_pose)
    direction_dist = np.linalg.norm(direction)
    if direction_dist <= 1e-9:
        return 0.0, 0.0
    direction_norm = direction / direction_dist
    speed=max_speed
    if costmap is not None:
      obstacle_y, obstacle_x = np.nonzero(costmap >= 254)
      if len(obstacle_x) > 0:
          obstacle_vectors = np.stack((obstacle_x - current_pose[0], obstacle_y - current_pose[1]), axis=1)
          obstacle_distance = np.linalg.norm(obstacle_vectors, axis=1)
          valid_distance = obstacle_distance > 1e-9
          cos_angle = np.zeros_like(obstacle_distance)
          cos_angle[valid_distance] = (obstacle_vectors[valid_distance] @ direction_norm) / obstacle_distance[valid_distance]
          cos_threshold = 0.75
          front_mask = cos_angle >= cos_threshold
          if np.any(front_mask):
              distance_from_obstacle = np.min(obstacle_distance[front_mask] / cos_angle[front_mask])
              speed = np.min([max_speed,np.sqrt(2*distance_from_obstacle*max_accel)])  # slow down if dangerous cells are ahead
    return tuple(direction_norm * speed*dist_to_goal/slow_radius if dist_to_goal<slow_radius else direction_norm * speed)
    # TODO: Implement Pure Pursuit controller.
