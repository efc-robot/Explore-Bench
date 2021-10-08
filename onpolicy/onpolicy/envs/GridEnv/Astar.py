"""
A_star 2D
@author: huiming zhou
"""

import os
import sys
import math
import heapq

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

# from Search_2D import plotting, env

import numpy as np

class AStar:
    """AStar set the cost + heuristics as the priority
    """
    def __init__(self, s_start, s_goal, map, heuristic_type):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type


        self.u_set = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                        (1, 0), (1, -1), (0, -1), (-1, -1)]  # feasible input set

        self.map = map  # costmap 

        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come

    def searching(self):
        """
        A_star Searching.
        :return: path, visited order
        """

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN,
                       (self.f_value(self.s_start), self.s_start))

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)

            if s == self.s_goal:  # stop condition
                break

            for s_n in self.get_neighbor(s):
                new_cost = self.g[s] + self.cost(s, s_n)

                if s_n not in self.g:
                    self.g[s_n] = math.inf

                if new_cost < self.g[s_n]:  # conditions for updating Cost
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

        return self.extract_path(self.PARENT), self.CLOSED

    def searching_repeated_astar(self, e):
        """
        repeated A*.
        :param e: weight of A*
        :return: path and visited order
        """

        path, visited = [], []

        while e >= 1:
            p_k, v_k = self.repeated_searching(self.s_start, self.s_goal, e)
            path.append(p_k)
            visited.append(v_k)
            e -= 0.5

        return path, visited

    def repeated_searching(self, s_start, s_goal, e):
        """
        run A* with weight e.
        :param s_start: starting state
        :param s_goal: goal state
        :param e: weight of a*
        :return: path and visited order.
        """

        g = {s_start: 0, s_goal: float("inf")}
        PARENT = {s_start: s_start}
        OPEN = []
        CLOSED = []
        heapq.heappush(OPEN,
                       (g[s_start] + e * self.heuristic(s_start), s_start))

        while OPEN:
            _, s = heapq.heappop(OPEN)
            CLOSED.append(s)

            if s == s_goal:
                break

            for s_n in self.get_neighbor(s):
                new_cost = g[s] + self.cost(s, s_n)

                if s_n not in g:
                    g[s_n] = math.inf

                if new_cost < g[s_n]:  # conditions for updating Cost
                    g[s_n] = new_cost
                    PARENT[s_n] = s
                    heapq.heappush(OPEN, (g[s_n] + e * self.heuristic(s_n), s_n))

        return self.extract_path(PARENT), CLOSED

    def get_neighbor(self, s):
        """
        find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """

        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        # if self.is_collision(s_start, s_goal):
        if self.is_collision(s_goal):
            return math.inf

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    # def is_collision(self, s_start, s_end):
    #     """
    #     check if the line segment (s_start, s_end) is collision.
    #     :param s_start: start node
    #     :param s_end: end node
    #     :return: True: is collision / False: not collision
    #     """

    #     if s_start in self.obs or s_end in self.obs:
    #         return True

    #     if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
    #         if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
    #             s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
    #             s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
    #         else:
    #             s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
    #             s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

    #         if s1 in self.obs or s2 in self.obs:
    #             return True

    #     return False

    def is_collision(self, s_end):
        """
        check if the line segment (s_start, s_end) is collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """
        if self.map[s_end[0], s_end[1]] == 0:
            return True

        return False

    def f_value(self, s):
        """
        f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """

        return self.g[s] + self.heuristic(s)

    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.s_goal]
        s = self.s_goal

        while True:
            s = PARENT[s]
            path.append(s)

            if s == self.s_start:
                break

        return list(path)

    def heuristic(self, s):
        """
        Calculate heuristic.
        :param s: current node (state)
        :return: heuristic function value
        """

        heuristic_type = self.heuristic_type  # heuristic type
        goal = self.s_goal  # goal node

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])

def obstacle_inflation(map, radius, resolution):
    inflation_grid = math.ceil(radius / resolution)
    import copy
    inflation_map = copy.deepcopy(map)
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if map[i][j] == 0:
                neighbor_list = get_neighbor(i, j, inflation_grid, map.shape[0], map.shape[1])
                for inflation_point in neighbor_list:
                    inflation_map[inflation_point[0],inflation_point[1]] = 0
    return inflation_map

def get_neighbor(x, y, radius, x_max, y_max):
    neighbor_list = []
    for i in range(-radius, radius+1):
        for j in range(-radius, radius+1):
            if x+i > -1 and x+i < x_max and y+j > -1 and y+j < y_max:
                neighbor_list.append([x+i,y+j])
    return neighbor_list

def plot_path(path, map):
    for pixel in path:
        map[pixel[0],pixel[1]] = 128
    return map


def main():
    # s_start = (60, 100)
    # s_goal = (230, 270)
    s_start = (290, 290)
    s_goal = (295, 310)

    from PIL import Image
    map_img = Image.open("/home/nics/catkin_ws/small_room_005.pgm")
    gt_map = np.array(map_img)
    inflation_map = obstacle_inflation(gt_map, 0.15, 0.05)
    # img = Image.fromarray(inflation_map.astype('uint8'))
    # img.save("inflation_map_005.pgm")

    astar = AStar(s_start, s_goal, inflation_map, "euclidean")
    # plot = plotting.Plotting(s_start, s_goal)

    path, visited = astar.searching()
    vis_map = plot_path(path, inflation_map)
    img = Image.fromarray(inflation_map.astype('uint8'))
    img.save("Astar.pgm")
    img.show()
    import pdb; pdb.set_trace()
    plot.animation(path, visited, "A*")  # animation

    # path, visited = astar.searching_repeated_astar(2.5)               # initial weight e = 2.5
    # plot.animation_ara_star(path, visited, "Repeated A*")


if __name__ == '__main__':
    main()
