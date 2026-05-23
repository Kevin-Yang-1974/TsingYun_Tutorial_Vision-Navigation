import numpy as np
import heapq
from typing import List, Tuple



class AStarPlanner:
    def __init__(self,costmap:np.ndarray):
        self.costmap = costmap
        self.rows, self.cols = costmap.shape
        self.actual_cost_map = np.full((self.rows, self.cols), np.inf)
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        #Euclidean distance
        alpha = 0.8
        return alpha*np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    def actual_cost(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return self.actual_cost_map[a[1], a[0]] + (np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) if a != b else 0)+self.costmap[b[1], b[0]]
    def get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (node[0] + dx, node[1] + dy)
                if not (0 <= neighbor[0] < self.cols and 0 <= neighbor[1] < self.rows):
                    continue
                if self.costmap[neighbor[1], neighbor[0]] >= 254:
                    continue
                if dx != 0 and dy != 0:
                    side_x = (node[0] + dx, node[1])
                    side_y = (node[0], node[1] + dy)
                    if self.costmap[side_x[1], side_x[0]] >= 254 or self.costmap[side_y[1], side_y[0]] >= 254:#may pass through corner of obstacle
                        continue
                neighbors.append(neighbor)
        return neighbors
    def reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]  # Return reversed path
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        self.actual_cost_map.fill(np.inf)
        start_node = (int(start[0]), int(start[1]))  
        goal_node = (int(goal[0]), int(goal[1]))  
        if  0<= start_node[0] < self.cols and 0 <= start_node[1] < self.rows and 0 <= goal_node[0] < self.cols and 0 <= goal_node[1] < self.rows:
            if self.costmap[start_node[1], start_node[0]] >= 254 or self.costmap[goal_node[1], goal_node[0]] >= 254:
                return []  # Start or goal is in a lethal cell

            open_set = []
            closed_set = set()
            heapq.heappush(open_set, (0, start_node))
            came_from = {}
            self.actual_cost_map[start_node[1], start_node[0]] = 0

            while open_set:
                current = heapq.heappop(open_set)[1]
                if current == goal_node:
                    return self.reconstruct_path(came_from, current)

                closed_set.add(current)
                for neighbor in self.get_neighbors(current):
                    if neighbor in closed_set:
                        continue
                    tentative_g_score = self.actual_cost(current, neighbor)

                    if tentative_g_score < self.actual_cost_map[neighbor[1], neighbor[0]]:
                        came_from[neighbor] = current
                        self.actual_cost_map[neighbor[1], neighbor[0]] = tentative_g_score
                        f_score = tentative_g_score + self.heuristic(neighbor, goal_node)
                        heapq.heappush(open_set, (f_score, neighbor))
                closed_set.add(current)
        return []  # No path found
