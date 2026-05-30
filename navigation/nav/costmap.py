"""Costmap generation: obstacle inflation and lidar-based dynamic costmap."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy

static_cost_map = None  

def compute_costmap(
    static_map: np.ndarray,
) -> np.ndarray:
    if not np.any(static_map == 1):
      return np.zeros_like(static_map, dtype=np.uint8)
    inflation_radius = 4.5 
    distance_map =(static_map == 0).astype(np.uint8)  
    distance_map=scipy.ndimage.distance_transform_edt(distance_map)
    cost_map =compute_cost(distance_map, 1.6, inflation_radius)
    return cost_map
    
    

def update_local_costmap(
    static_map: np.ndarray,
    robot_pos: Tuple[float, float],
    lidar_scan: np.ndarray,
    lidar_range: float,
    lidar_num_rays: int,
) -> np.ndarray:
   
    global static_cost_map
    dynamic_inflation_radius = 6.0 
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
    mask=(dynamic_map==static_map)
    dynamic_map[mask]=0
    if not np.any(dynamic_map == 1):
      return static_cost_map
    distance_map =(dynamic_map == 0).astype(np.uint8)  
    distance_map=scipy.ndimage.distance_transform_edt(distance_map)
    dynamic_cost_map = compute_cost(distance_map, 1.6, dynamic_inflation_radius)
    return np.maximum(static_cost_map, dynamic_cost_map)
    
def compute_cost(distance_map: np.ndarray, r_inscribed:float,r_inflation:float) -> np.ndarray:
    alpha=np.log(253/128)/(r_inflation-2*r_inscribed)#scaling factor for exponential decay(r_inflation>2*r_inscribed)
    cost_map=np.zeros_like(distance_map, dtype=np.uint8)
    cost_map[distance_map ==0] = 255
    cost_map[(distance_map > 0) & (distance_map <= r_inscribed)]=254
    mask=(distance_map >r_inscribed)&(distance_map <=r_inflation)
    cost_map[mask]=np.round(253*np.exp(-alpha*(distance_map[mask]-r_inscribed))).astype(np.uint8)
    return cost_map