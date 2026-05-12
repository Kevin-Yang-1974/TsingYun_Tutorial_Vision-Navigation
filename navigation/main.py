"""
Navigation demo entry point.

Run from the project root:
    python navigation/main.py --level 1 --record output.mp4

Controls
--------
Left click  : set a debug goal (timer frozen)
Right click : reset robot, dynamic obstacles, and timer (back to timed mode)
ESC / close : quit
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pygame
import yaml

from core.recorder import Recorder
from core.renderer import Renderer
from core.robot import Robot
from core.sensor import Lidar
from core.task import Task
from core.world import DEFAULT_MAP, MapWorld
from nav.controller import local_plan
from nav.costmap import compute_costmap, update_local_costmap
from nav.planner import global_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="2D Navigation Simulator")
    parser.add_argument(
        "--level", "-l",
        type=int, choices=(1, 2), default=1,
        help="level number (1: static obstacles, 2: add dynamic obstacles and lidar sensing)",
    )
    parser.add_argument(
        "--record", "-r",
        default="output.mp4", metavar="FILE",
        help="record session to video file, e.g. --record out.mp4",
    )
    args = parser.parse_args()

    level: int = args.level

    config_path = os.path.join(os.path.dirname(__file__), "configs", "cfg.yaml")
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    fps: int = int(cfg["fps"])

    robot = Robot(
        radius=float(cfg["robot"]["radius"]),
        max_speed=float(cfg["robot"]["max_speed"]),
        max_accel=float(cfg["robot"]["max_accel"]),
    )

    world = MapWorld(
        DEFAULT_MAP,
        robot=robot,
        enable_dynamic=(level >= 2),
        dynamic_radius=float(cfg["world"]["dynamic_radius"]),
        dynamic_speed_range=tuple(cfg["world"]["dynamic_speed_range"]),
        dynamic_bboxes=cfg["world"]["dynamic_bboxes"],
        rng_seed=int(cfg["world"]["rng_seed"]),
    )

    task = Task(
        world=world,
        start_pos=tuple(cfg["task"]["start_pos"]),
        goal_pos=tuple(cfg["task"]["goal_pos"]),
        goal_threshold=float(cfg["task"]["goal_threshold"]),
        pos_eps=float(cfg["task"]["pos_eps"]),
        stable_frames=int(cfg["task"]["stable_frames"]),
    )

    lidar = Lidar(
        max_range=float(cfg["lidar"]["range"]),
        num_rays=int(cfg["lidar"]["rays"]),
    )

    renderer = Renderer(
        level=level,
        task=task,
        lidar=lidar,
        cell_size=int(cfg["renderer"]["cell_size"]),
        sidebar_width=int(cfg["renderer"]["sidebar_width"]),
    )

    recorder = Recorder(renderer.screen, args.record, fps)

    if args.record:
        recorder.start()

    clock = pygame.time.Clock()
    global_path = []

    active_costmap = compute_costmap(world.static_map)

    running = True
    while running:
        dt = min(clock.tick(fps) / 1000.0, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = event.pos
                    if renderer.in_map(mx, my):
                        gx, gy = renderer.s2w(mx, my)
                        gr, gc = int(gy), int(gx)
                        if 0 <= gr < world.rows and 0 <= gc < world.cols and world.static_map[gr, gc] == 0:
                            task.set_debug_goal(gx, gy)
                            global_path = []
                elif event.button == 3:
                    task.reset()
                    global_path = []

        robot_pos = robot.get_pose()

        if level >= 2:
            lidar.scan(world)
            active_costmap = update_local_costmap(world.static_map, robot_pos, lidar.distances, lidar.max_range, lidar.num_rays)

        global_path = global_plan(robot_pos, task.goal, active_costmap)

        if global_path:
            cmd_vx, cmd_vy = local_plan(robot_pos, robot.max_speed, robot.max_accel, global_path, active_costmap)
        else:
            cmd_vx, cmd_vy = 0.0, 0.0

        task.update(cmd_vx, cmd_vy, global_path, dt)
        renderer.render(global_path, active_costmap)
        recorder.capture()

    recorder.stop()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
