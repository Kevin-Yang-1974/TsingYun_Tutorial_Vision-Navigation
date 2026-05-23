"""Costmap generation: obstacle inflation and lidar-based dynamic costmap."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy

static_cost_map = None  

def compute_costmap(
    static_map: np.ndarray,
) -> np.ndarray:
    """
    Build the global costmap by inflating static obstacles.

    Parameters
    ----------
    static_map : np.ndarray, shape (rows, cols), dtype int8
        0 = free cell, 1 = obstacle cell.

    Returns
    -------
    costmap : np.ndarray, shape (rows, cols), dtype uint8
        Per-cell cost in [0, 255]:
        - obstacle cells get the maximum lethal value, so the planner
          treats them as impassable.
        - free cells near an obstacle get a non-zero cost that decays with
          distance, creating a "buffer" so the planned path keeps clear of
          walls instead of grazing them.
        - free cells far from any obstacle get cost 0.

    Notes
    -----
    - The classical recipe: compute the Euclidean distance from each free cell
      to the nearest obstacle (`scipy.ndimage.distance_transform_edt` does this
      in one call), then map distance → cost so that distance 0 is lethal and
      cost falls off smoothly out to some `inflation_radius`. Beyond that
      radius, cost should be 0.
    - The shape of the decay (linear, exponential, ...) and the magnitude of
      the inflation radius are tuning knobs. Pick something that visibly biases
      the path away from walls without making narrow passages impassable. The
      inflation radius that is too large will also cause the robot to take a
      longer route, wasting time.
    """
    if not np.any(static_map == 1):
      return np.zeros_like(static_map, dtype=np.uint8)
    inflation_radius = 4.5  
    distance_map =(static_map == 0).astype(np.uint8)  
    distance_map=scipy.ndimage.distance_transform_edt(distance_map)
    cost_map =compute_cost(distance_map, 1.6, inflation_radius)
    return cost_map
    # TODO: Implement a function to compute a costmap from the static map by inflating obstacles.
    

def update_local_costmap(
    static_map: np.ndarray,
    robot_pos: Tuple[float, float],
    lidar_scan: np.ndarray,
    lidar_range: float,
    lidar_num_rays: int,
) -> np.ndarray:
    """
    Produce the per-frame costmap by adding a dynamic layer on top of the
    static inflation.

    Parameters
    ----------
    static_map : np.ndarray, shape (rows, cols), dtype int8
        The same static map passed to `compute_costmap`. Re-inflate it (or
        cache the result) to get the static layer.
    robot_pos : Tuple[float, float], (x, y)
        Current robot position in world (grid-unit) coordinates. Lidar rays
        originate from this point.
    lidar_scan : np.ndarray, shape (lidar_num_rays,)
        Hit distance for each ray, in grid units. A value equal to `lidar_range`
        means the ray did not hit anything within range.
    lidar_range : float
        Maximum sensing distance of the lidar, in grid units.
    lidar_num_rays : int
        Number of rays in the scan; the i-th ray is at angle
        `2*pi * i / lidar_num_rays` measured from the +x axis.

    Returns
    -------
    costmap : np.ndarray, shape (rows, cols), dtype uint8
        Static-inflation layer merged with a dynamic layer that marks lidar
        hits as lethal and inflates them with a (smaller) buffer. Use a
        per-cell `max` to combine the two layers so the most conservative
        cost wins.

    Notes
    -----
    - Convert each ray hit `(angle_i, lidar_scan[i])` into a world point
      `(x + d*cos(a), y + d*sin(a))`, then to a grid cell. Mark that cell
      lethal and inflate it.
    - Skip rays where `lidar_scan[i] >= lidar_range` (no hit).
    - Optional but useful: skip hits that land on a cell that is *already*
      a static obstacle; otherwise the lidar's view of a wall keeps
      re-inflating the same area.
    """
    global static_cost_map
    dynamic_inflation_radius = 4.0  
    if static_cost_map is None:
        static_cost_map = compute_costmap(static_map)
    angles = np.linspace(0, 2 * np.pi, lidar_num_rays, endpoint=False)
    hit=np.stack((lidar_scan,angles),axis=1)
    valid_hits=hit[hit[:,0]<lidar_range]
    hit_points=np.zeros((valid_hits.shape[0],2))
    hit_points[:,0]=valid_hits[:,0]*np.cos(valid_hits[:,1])+robot_pos[0]
    hit_points[:,1]=valid_hits[:,0]*np.sin(valid_hits[:,1])+robot_pos[1]
    rows,cols=static_map.shape
    hit_points=hit_points.astype(np.int32)
    mask=(hit_points[:,0]>=0)&(hit_points[:,0]<cols)&(hit_points[:,1]>=0)&(hit_points[:,1]<rows)
    hit_points=hit_points[mask]
    dynamic_map=np.zeros_like(static_map)
    dynamic_map[hit_points[:,1].astype(np.int32),hit_points[:,0].astype(np.int32)]=1
    if not np.any(dynamic_map == 1):
      return static_cost_map
    distance_map =(dynamic_map == 0).astype(np.uint8)  
    distance_map=scipy.ndimage.distance_transform_edt(distance_map)
    dynamic_cost_map = compute_cost(distance_map, 1.6, dynamic_inflation_radius)
    return np.maximum(static_cost_map, dynamic_cost_map)
    # TODO: Implement a function to update the global costmap with a local dynamic layer based on the lidar scan.
    
def compute_cost(distance_map: np.ndarray, r_inscribed:float,r_inflation:float) -> np.ndarray:
    alpha=np.log(253/128)/(r_inflation-2*r_inscribed)#scaling factor for exponential decay(r_inflation>2*r_inscribed)
    cost_map=np.zeros_like(distance_map, dtype=np.uint8)
    cost_map[distance_map ==0] = 255
    cost_map[(distance_map > 0) & (distance_map <= r_inscribed)]=254
    mask=(distance_map >r_inscribed)&(distance_map <=r_inflation)
    cost_map[mask]=np.round(253*np.exp(-alpha*(distance_map[mask]-r_inscribed))).astype(np.uint8)
    return cost_map