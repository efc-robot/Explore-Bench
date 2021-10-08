from PIL import Image
import numpy as np
import copy
import math
import time
from .window import Window
# from window import Window
import gym
import sys
from .Astar import AStar
# from Astar import AStar
import random
import os

class GridMap(gym.Env):
    def __init__(self, origin, data):
        self.origin = origin
        self.data = data

class GazeboEnv(gym.Env):
    def __init__(self, resolution, sensor_range, num_agents, max_steps,
        use_merge = True,
        use_same_location = True,
        use_complete_reward = True,
        use_multiroom = False,
        use_time_penalty = False,
        use_single_reward = False,
        visualization = False):

        self.num_agents = num_agents

        # map_img = Image.open(map_file)
        # self.gt_map = np.array(map_img)
        # self.inflation_map = obstacle_inflation(self.gt_map, 0.15, 0.05)
        self.resolution = resolution
        self.sensor_range = sensor_range
        self.pos_traj = []
        self.grid_traj = []
        self.map_per_frame = []
        self.robot = Robot([2,2], 0)
        self.step_time = 0.1
        self.dw_time = 1
        self.minDis2Frontier = 2*self.resolution
        self.frontiers = []
        self.path_log = []
        for e in range(self.num_agents):
            self.path_log.append([])
        self.built_map = []

        # self.width = self.gt_map.shape[1]
        # self.height = self.gt_map.shape[0]

        self.resize_width = 64
        self.resize_height = 64

        self.robot_discrete_dir = [i*math.pi for i in range(16)]
        self.agent_view_size = int(sensor_range/self.resolution)
        self.target_ratio = 0.98
        self.merge_ratio = 0
        self.merge_reward = 0
        self.agent_reward = np.zeros((num_agents))
        self.agent_ratio_step = np.ones((num_agents)) * max_steps
        self.merge_ratio_step = max_steps
        self.max_steps = max_steps
        # self.total_cell_size = np.sum((self.gt_map != 205).astype(int))

        self.use_same_location = use_same_location
        self.use_complete_reward = use_complete_reward
        self.use_multiroom = use_multiroom
        self.use_time_penalty = use_time_penalty
        self.use_merge = use_merge
        self.use_single_reward = use_single_reward

        # define space
        self.action_space = [gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32) for _ in range(self.num_agents)]

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        global_observation_space = {}
        global_observation_space['global_obs'] = gym.spaces.Box(
            low=0, high=255, shape=(4, self.resize_width, self.resize_height), dtype='uint8')
        
        #global_observation_space['global_merge_goal'] = gym.spaces.Box(
            #low=0, high=255, shape=(2, self.width, self.height), dtype='uint8')

        # global_observation_space['image'] = gym.spaces.Box(
        #     low=0, high=255, shape=(self.resize_width, self.resize_height, 3), dtype='uint8')

        # global_observation_space['vector'] = gym.spaces.Box(
        #     low=-1, high=1, shape=(self.num_agents,), dtype='float')
        global_observation_space['vector'] = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype='float')
        if use_merge:
            global_observation_space['global_merge_obs'] = gym.spaces.Box(
                low=0, high=255, shape=(4, self.resize_width, self.resize_height), dtype='uint8')
            # global_observation_space['global_direction'] = gym.spaces.Box(
            #     low=-1, high=1, shape=(self.num_agents, 4), dtype='float')
        # else:
        #     global_observation_space['global_direction'] = gym.spaces.Box(
        #         low=-1, high=1, shape=(1, 4), dtype='float')
        share_global_observation_space = global_observation_space.copy()
        # share_global_observation_space['gt_map'] = gym.spaces.Box(
        #     low=0, high=255, shape=(1, self.width, self.height), dtype='uint8')
        
        global_observation_space = gym.spaces.Dict(global_observation_space)
        share_global_observation_space = gym.spaces.Dict(share_global_observation_space)

        self.observation_space = []
        self.share_observation_space = []

        for agent_id in range(self.num_agents):
            self.observation_space.append(global_observation_space)
            self.share_observation_space.append(share_global_observation_space)

        self.visualization = visualization
        if self.visualization:
            self.window = Window('map')
            self.window.show(block=False)

        # self.visualize_map = np.zeros((self.width, self.height))
        self.visualize_goal = [[0,0] for i in range(self.num_agents)]
       
    def reset(self):
        # 1. read from blueprints files randomly
        map_file = random.choice(os.listdir('/home/nics/git_ws/src/onpolicy/onpolicy/envs/Gazebo2D/datasets'))
        map_img = Image.open(os.path.join('/home/nics/git_ws/src/onpolicy/onpolicy/envs/Gazebo2D/datasets', map_file))
        # map_img = Image.open('/home/nics/workspace/blueprints/room1_modified.pgm')
        self.gt_map = np.array(map_img)
        self.inflation_map = obstacle_inflation(self.gt_map, 0.15, 0.05)
        self.width = self.gt_map.shape[1]
        self.height = self.gt_map.shape[0]
        self.total_cell_size = np.sum((self.gt_map != 205).astype(int))
        self.visualize_map = np.zeros((self.width, self.height))

        self.num_step = 0
        obs = []
        self.built_map = []

        # reset robot pos and dir
        # self.agent_pos = [self.continuous_to_discrete([-8,8])]
        # self.agent_dir = [0]
        self.agent_pos = []
        self.agent_dir = []

        for i in range(self.num_agents):
            random_at_obstacle_or_unknown = True
            while(random_at_obstacle_or_unknown):
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                if self.gt_map[x][y] == 254:     # free space
                    self.agent_pos.append([x, y])
                    # self.agent_dir.append(random.randint(0, 15)*math.pi)
                    self.agent_dir.append(random.randint(0, 3))
                    random_at_obstacle_or_unknown = False

        # init local map
        self.explored_each_map = []
        self.obstacle_each_map = []
        self.previous_explored_each_map = []
        current_agent_pos = []

        for i in range(self.num_agents):
            self.explored_each_map.append(np.zeros((self.width, self.height)))
            self.obstacle_each_map.append(np.zeros((self.width, self.height)))
            self.previous_explored_each_map.append(np.zeros((self.width, self.height)))

        for i in range(self.num_agents):
            _, map_this_frame, _, _ = self.optimized_build_map(self.discrete_to_continuous(self.agent_pos[i]), 0, self.gt_map, self.resolution, self.sensor_range)
            # unknown: 205   free: 254   occupied: 0
            self.built_map.append(map_this_frame)
            obs.append(map_this_frame)
            current_agent_pos.append(self.agent_pos[i])
            self.explored_each_map[i] = (map_this_frame != 205).astype(int)
            self.obstacle_each_map[i] = (map_this_frame == 0).astype(int)

        explored_all_map = np.zeros((self.width, self.height))
        obstacle_all_map = np.zeros((self.width, self.height))
        self.previous_all_map = np.zeros((self.width, self.height))
        for i in range(self.num_agents):
            explored_all_map += self.explored_each_map[i]
            obstacle_all_map += self.obstacle_each_map[i]    
        explored_all_map = (explored_all_map > 0).astype(int)
        obstacle_all_map = (obstacle_all_map > 0).astype(int)

        # if we have both explored map and obstacle map, we can merge them to get complete map
        # obstacle: 2   free: 1   unknown: 0
        temp = explored_all_map + obstacle_all_map
        self.complete_map = np.zeros(temp.shape)
        self.complete_map[temp == 2] = 0
        self.complete_map[temp == 1] = 254
        self.complete_map[temp == 0] = 205

        info = {}
        info['explored_all_map'] = np.array(explored_all_map)
        info['current_agent_pos'] = np.array(current_agent_pos)
        info['explored_each_map'] = np.array(self.explored_each_map)
        info['obstacle_all_map'] = np.array(obstacle_all_map)
        info['obstacle_each_map'] = np.array(self.obstacle_each_map)
        info['agent_direction'] = np.array(self.agent_dir)
        # info['agent_local_map'] = self.agent_local_map

        info['merge_explored_ratio'] = self.merge_ratio
        info['merge_explored_reward'] = self.merge_reward
        info['agent_explored_reward'] = self.agent_reward
        info['merge_ratio_step'] = self.merge_ratio_step

        for i in range(self.num_agents):
            info["agent{}_ratio_step".format(i)] = self.agent_ratio_step[i]

        self.merge_ratio = 0
        self.merge_reward = 0
        self.agent_reward = np.zeros((self.num_agents))
        self.agent_ratio_step = np.ones((self.num_agents)) * self.max_steps
        self.merge_ratio_step = self.max_steps

        obs = np.array(obs)
        if self.visualization:
            # self.window.show_img(self.built_map[0])
            self.window.show_img(self.complete_map)
        return obs, info

    def reset_for_traditional(self):
        # 1. read from blueprints files randomly
        # map_file = random.choice(os.listdir('/home/nics/workspace/blueprints'))
        # map_img = Image.open(os.path.join('/home/nics/workspace/blueprints', map_file))
        map_img = Image.open('/home/nics/workspace/blueprints/room1_modified.pgm')
        self.gt_map = np.array(map_img)
        # self.window.show_img(self.gt_map)
        # import pdb; pdb.set_trace()
        self.inflation_map = obstacle_inflation(self.gt_map, 0.15, 0.05)
        self.width = self.gt_map.shape[1]
        self.height = self.gt_map.shape[0]
        self.total_cell_size = np.sum((self.gt_map != 205).astype(int))
        self.visualize_map = np.zeros((self.width, self.height))

        self.num_step = 0
        obs = []
        self.built_map = []
        # reset robot pos and dir
        # self.agent_pos = [self.continuous_to_discrete([-8, 8.2]), self.continuous_to_discrete([8, -8])]
        self.agent_pos = [self.continuous_to_discrete([-8, 8])] # 7.2
        self.agent_dir = [0]

        # for i in range(self.num_agents):
        #     random_at_obstacle_or_unknown = True
        #     while(random_at_obstacle_or_unknown):
        #         x = random.randint(0, self.width - 1)
        #         y = random.randint(0, self.height - 1)
        #         if self.gt_map[x][y] == 254:     # free space
        #             self.agent_pos.append([x, y])
        #             # self.agent_dir.append(random.randint(0, 15)*math.pi)
        #             self.agent_dir.append(random.randint(0, 3))
        #             random_at_obstacle_or_unknown = False

        # init local map
        self.explored_each_map = []
        self.obstacle_each_map = []
        self.previous_explored_each_map = []
        current_agent_pos = []

        for i in range(self.num_agents):
            self.explored_each_map.append(np.zeros((self.width, self.height)))
            self.obstacle_each_map.append(np.zeros((self.width, self.height)))
            self.previous_explored_each_map.append(np.zeros((self.width, self.height)))

        for i in range(self.num_agents):
            _, map_this_frame, _, _ = self.optimized_build_map(self.discrete_to_continuous(self.agent_pos[i]), 0, self.gt_map, self.resolution, self.sensor_range)
            # unknown: 205   free: 254   occupied: 0
            self.built_map.append(map_this_frame)
            obs.append(map_this_frame)
            current_agent_pos.append(self.agent_pos[i])
            self.explored_each_map[i] = (map_this_frame != 205).astype(int)
            self.obstacle_each_map[i] = (map_this_frame == 0).astype(int)

        explored_all_map = np.zeros((self.width, self.height))
        obstacle_all_map = np.zeros((self.width, self.height))
        self.previous_all_map = np.zeros((self.width, self.height))
        for i in range(self.num_agents):
            explored_all_map += self.explored_each_map[i]
            obstacle_all_map += self.obstacle_each_map[i]    
        explored_all_map = (explored_all_map > 0).astype(int)
        obstacle_all_map = (obstacle_all_map > 0).astype(int)

        # if we have both explored map and obstacle map, we can merge them to get complete map
        # obstacle: 2   free: 1   unknown: 0
        temp = explored_all_map + obstacle_all_map
        self.complete_map = np.zeros(temp.shape)
        self.complete_map[temp == 2] = 0
        self.complete_map[temp == 1] = 254
        self.complete_map[temp == 0] = 205

        info = {}
        info['explored_all_map'] = np.array(explored_all_map)
        info['current_agent_pos'] = np.array(current_agent_pos)
        info['explored_each_map'] = np.array(self.explored_each_map)
        info['obstacle_all_map'] = np.array(obstacle_all_map)
        info['obstacle_each_map'] = np.array(self.obstacle_each_map)
        info['agent_direction'] = np.array(self.agent_dir)
        # info['agent_local_map'] = self.agent_local_map

        info['merge_explored_ratio'] = self.merge_ratio
        info['merge_explored_reward'] = self.merge_reward
        info['agent_explored_reward'] = self.agent_reward
        info['merge_ratio_step'] = self.merge_ratio_step

        for i in range(self.num_agents):
            info["agent{}_ratio_step".format(i)] = self.agent_ratio_step[i]

        self.merge_ratio = 0
        self.merge_reward = 0
        self.agent_reward = np.zeros((self.num_agents))
        self.agent_ratio_step = np.ones((self.num_agents)) * self.max_steps
        self.merge_ratio_step = self.max_steps

        obs = np.array(obs)
        if self.visualization:
            # self.window.show_img(self.built_map[0])
            self.window.show_img(self.complete_map)
        return obs, info

    def step(self, action):
        obs = []
        flag = False
        self.explored_each_map_t = []
        self.obstacle_each_map_t = []
        current_agent_pos = []
        each_agent_rewards = []
        self.num_step += 1
        reward_obstacle_each_map = np.zeros((self.num_agents, self.width, self.height))
        delta_reward_each_map = np.zeros((self.num_agents, self.width, self.height))
        reward_explored_each_map = np.zeros((self.num_agents, self.width, self.height))
        explored_all_map = np.zeros((self.width, self.height))
        obstacle_all_map = np.zeros((self.width, self.height))

        for i in range(self.num_agents):
            self.explored_each_map_t.append(np.zeros((self.width, self.height)))
            self.obstacle_each_map_t.append(np.zeros((self.width, self.height)))
        for i in range(self.num_agents): 
            robotGoal = action[i]
            if robotGoal[0] == self.agent_pos[i][0] and robotGoal[1] == self.agent_pos[i][1]:
                print("finish exploration")
                flag = True
                pass
            else:
                if self.gt_map[robotGoal[0], robotGoal[1]] == 254 and self.gt_map[self.agent_pos[i][0], self.agent_pos[i][1]] == 254:
                    global_plan = self.Astar_global_planner(self.agent_pos[i], robotGoal)   
                    pose = self.naive_local_planner(global_plan)
                    self.path_log[0].extend(pose)
                    self.agent_pos[i] = pose[-1][0]
                    # self.agent_dir[i] = pose[-1][1]
                    self.agent_dir[i] = random.randint(0, 3)
                    self.build_map_given_path_for_multi_robot(pose, i)
                else:
                    print("Choose a non-free frontier")

        for i in range(self.num_agents): 
            # _, map_this_frame, _, _ = self.optimized_build_map(self.discrete_to_continuous(self.agent_pos[i]), 0, self.gt_map, self.resolution, self.sensor_range)
            # unknown: 205   free: 254   occupied: 0
            obs.append(self.built_map[i])
            current_agent_pos.append(self.agent_pos[i])
            self.explored_each_map_t[i] = (self.built_map[i] != 205).astype(int)
            self.obstacle_each_map_t[i] = (self.built_map[i] == 0).astype(int)

        for i in range(self.num_agents):
            self.explored_each_map[i] = np.maximum(self.explored_each_map[i], self.explored_each_map_t[i])
            self.obstacle_each_map[i] = np.maximum(self.obstacle_each_map[i], self.obstacle_each_map_t[i])
           
            reward_explored_each_map[i] = self.explored_each_map[i].copy()
            reward_explored_each_map[i][reward_explored_each_map[i] != 0] = 1
            
            reward_previous_explored_each_map = self.previous_explored_each_map[i].copy()
            reward_previous_explored_each_map[reward_previous_explored_each_map != 0] = 1

            # reward_obstacle_each_map[i] = self.obstacle_each_map[i].copy()
            # reward_obstacle_each_map[i][reward_obstacle_each_map[i] != 0] = 1

            delta_reward_each_map[i] = reward_explored_each_map[i]
            
            each_agent_rewards.append((np.array(delta_reward_each_map[i]) - np.array(reward_previous_explored_each_map)).sum())
            self.previous_explored_each_map[i] = self.explored_each_map[i]
        
        for i in range(self.num_agents):
            explored_all_map = np.maximum(self.explored_each_map[i], explored_all_map)
            obstacle_all_map = np.maximum(self.obstacle_each_map[i], obstacle_all_map)

        temp = explored_all_map + obstacle_all_map
        self.complete_map = np.zeros(temp.shape)
        self.complete_map[temp == 2] = 0
        self.complete_map[temp == 1] = 254
        self.complete_map[temp == 0] = 205

        reward_explored_all_map = explored_all_map.copy()
        reward_explored_all_map[reward_explored_all_map != 0] = 1

        delta_reward_all_map = reward_explored_all_map

        reward_previous_all_map = self.previous_all_map.copy()
        reward_previous_all_map[reward_previous_all_map != 0] = 1

        merge_explored_reward = (np.array(delta_reward_all_map) - np.array(reward_previous_all_map)).sum()
        self.previous_all_map = explored_all_map

        info = {}
        info['explored_all_map'] = np.array(explored_all_map)
        info['current_agent_pos'] = np.array(current_agent_pos)
        info['explored_each_map'] = np.array(self.explored_each_map)
        info['obstacle_all_map'] = np.array(obstacle_all_map)
        info['obstacle_each_map'] = np.array(self.obstacle_each_map)
        info['agent_direction'] = np.array(self.agent_dir)
        # info['agent_local_map'] = self.agent_local_map
        if self.use_time_penalty:
            info['agent_explored_reward'] = np.array(each_agent_rewards) * 0.02 - 0.01
            info['merge_explored_reward'] = merge_explored_reward * 0.02 - 0.01
        else:
            info['agent_explored_reward'] = np.array(each_agent_rewards) * 0.02
            info['merge_explored_reward'] = merge_explored_reward * 0.02
        done = False
        if delta_reward_all_map.sum() / self.total_cell_size >= self.target_ratio or flag:#(self.width * self.height)
            done = True  
            # save trajectory for visualization
            # import pickle
            # traj_file = open('rl_traj.pickle', 'w')
            # pickle.dump(self.path_log, traj_file) 
            # print("save successfully")    
            self.merge_ratio_step = self.num_step
            if self.use_complete_reward:
                info['merge_explored_reward'] += 0.1 * (delta_reward_all_map.sum() / self.total_cell_size)     
                
        for i in range(self.num_agents):
            if delta_reward_each_map[i].sum() / self.total_cell_size >= self.target_ratio:#(self.width * self.height)
                self.agent_ratio_step[i] = self.num_step
                # if self.use_complete_reward:
                #     info['agent_explored_reward'][i] += 0.1 * (reward_explored_each_map[i].sum() / (self.width * self.height))
        
        self.agent_reward = info['agent_explored_reward']
        self.merge_reward = info['merge_explored_reward']
        self.merge_ratio = delta_reward_all_map.sum() / self.total_cell_size #(self.width * self.height)
        info['merge_explored_ratio'] = self.merge_ratio
        info['merge_ratio_step'] = self.merge_ratio_step
        for i in range(self.num_agents):
            info["agent{}_ratio_step".format(i)] = self.agent_ratio_step[i]

        dones = np.array([done for agent_id in range(self.num_agents)])
        if self.use_single_reward:
            rewards = 0.3 * np.expand_dims(info['agent_explored_reward'], axis=1) + 0.7 * np.expand_dims(np.array([info['merge_explored_reward'] for _ in range(self.num_agents)]), axis=1)
        else:
            rewards = np.expand_dims(np.array([info['merge_explored_reward'] for _ in range(self.num_agents)]), axis=1)

        obs = np.array(obs)

        # self.plot_map_with_path()

        return obs, rewards, dones, info

    def step_each_grid(self, action):
        obs = []
        flag = False
        self.explored_each_map_t = []
        self.obstacle_each_map_t = []
        current_agent_pos = []
        each_agent_rewards = []
        self.num_step += 1
        reward_obstacle_each_map = np.zeros((self.num_agents, self.width, self.height))
        delta_reward_each_map = np.zeros((self.num_agents, self.width, self.height))
        reward_explored_each_map = np.zeros((self.num_agents, self.width, self.height))
        explored_all_map = np.zeros((self.width, self.height))
        obstacle_all_map = np.zeros((self.width, self.height))

        for i in range(self.num_agents):
            self.explored_each_map_t.append(np.zeros((self.width, self.height)))
            self.obstacle_each_map_t.append(np.zeros((self.width, self.height)))
        for i in range(self.num_agents): 
            robotGoal = action[i]
            if robotGoal[0] == self.agent_pos[i][0] and robotGoal[1] == self.agent_pos[i][1]:
                print("finish exploration")
                flag = True
                pass
            else:
                if self.gt_map[robotGoal[0], robotGoal[1]] == 254 and self.gt_map[self.agent_pos[i][0], self.agent_pos[i][1]] == 254:
                    global_plan = self.Astar_global_planner(self.agent_pos[i], robotGoal)   
                    pose = self.naive_local_planner(global_plan)
                    self.agent_pos[i] = pose[1][0]
                    self.path_log[i].append(self.agent_pos[i])
                    # import pdb; pdb.set_trace()
                    # self.agent_pos[i] = pose[-1][0]
                    # self.agent_dir[i] = pose[-1][1]
                    print("step each grid !!!!!!")
                    self.agent_dir[i] = random.randint(0, 3)
                    _, map_this_frame, (x_min, x_max), (y_min, y_max) = self.optimized_build_map(self.discrete_to_continuous(self.agent_pos[i]), 0, self.gt_map, self.resolution, self.sensor_range)  # can be modified to replace self.gt_map, self.resolution and self.sensor_range
                    self.built_map[i] = self.merge_two_map(self.built_map[i], map_this_frame, [x_min, x_max], [y_min, y_max])
                else:
                    print("Choose a non-free frontier")

        for i in range(self.num_agents): 
            # _, map_this_frame, _, _ = self.optimized_build_map(self.discrete_to_continuous(self.agent_pos[i]), 0, self.gt_map, self.resolution, self.sensor_range)
            # unknown: 205   free: 254   occupied: 0
            obs.append(self.built_map[i])
            current_agent_pos.append(self.agent_pos[i])
            self.explored_each_map_t[i] = (self.built_map[i] != 205).astype(int)
            self.obstacle_each_map_t[i] = (self.built_map[i] == 0).astype(int)

        for i in range(self.num_agents):
            self.explored_each_map[i] = np.maximum(self.explored_each_map[i], self.explored_each_map_t[i])
            self.obstacle_each_map[i] = np.maximum(self.obstacle_each_map[i], self.obstacle_each_map_t[i])
           
            reward_explored_each_map[i] = self.explored_each_map[i].copy()
            reward_explored_each_map[i][reward_explored_each_map[i] != 0] = 1
            
            reward_previous_explored_each_map = self.previous_explored_each_map[i].copy()
            reward_previous_explored_each_map[reward_previous_explored_each_map != 0] = 1

            # reward_obstacle_each_map[i] = self.obstacle_each_map[i].copy()
            # reward_obstacle_each_map[i][reward_obstacle_each_map[i] != 0] = 1

            delta_reward_each_map[i] = reward_explored_each_map[i]
            
            each_agent_rewards.append((np.array(delta_reward_each_map[i]) - np.array(reward_previous_explored_each_map)).sum())
            self.previous_explored_each_map[i] = self.explored_each_map[i]
        
        for i in range(self.num_agents):
            explored_all_map = np.maximum(self.explored_each_map[i], explored_all_map)
            obstacle_all_map = np.maximum(self.obstacle_each_map[i], obstacle_all_map)

        temp = explored_all_map + obstacle_all_map
        self.complete_map = np.zeros(temp.shape)
        self.complete_map[temp == 2] = 0
        self.complete_map[temp == 1] = 254
        self.complete_map[temp == 0] = 205

        reward_explored_all_map = explored_all_map.copy()
        reward_explored_all_map[reward_explored_all_map != 0] = 1

        delta_reward_all_map = reward_explored_all_map

        reward_previous_all_map = self.previous_all_map.copy()
        reward_previous_all_map[reward_previous_all_map != 0] = 1

        merge_explored_reward = (np.array(delta_reward_all_map) - np.array(reward_previous_all_map)).sum()
        self.previous_all_map = explored_all_map

        info = {}
        info['explored_all_map'] = np.array(explored_all_map)
        info['current_agent_pos'] = np.array(current_agent_pos)
        info['explored_each_map'] = np.array(self.explored_each_map)
        info['obstacle_all_map'] = np.array(obstacle_all_map)
        info['obstacle_each_map'] = np.array(self.obstacle_each_map)
        info['agent_direction'] = np.array(self.agent_dir)
        # info['agent_local_map'] = self.agent_local_map
        if self.use_time_penalty:
            info['agent_explored_reward'] = np.array(each_agent_rewards) * 0.02 - 0.01
            info['merge_explored_reward'] = merge_explored_reward * 0.02 - 0.01
        else:
            info['agent_explored_reward'] = np.array(each_agent_rewards) * 0.02
            info['merge_explored_reward'] = merge_explored_reward * 0.02
        done = False
        if delta_reward_all_map.sum() / self.total_cell_size >= self.target_ratio or flag:#(self.width * self.height)
            done = True       
            self.merge_ratio_step = self.num_step
            if self.use_complete_reward:
                info['merge_explored_reward'] += 0.1 * (delta_reward_all_map.sum() / self.total_cell_size)     
                
        for i in range(self.num_agents):
            if delta_reward_each_map[i].sum() / self.total_cell_size >= self.target_ratio:#(self.width * self.height)
                self.agent_ratio_step[i] = self.num_step
                # if self.use_complete_reward:
                #     info['agent_explored_reward'][i] += 0.1 * (reward_explored_each_map[i].sum() / (self.width * self.height))
        
        self.agent_reward = info['agent_explored_reward']
        self.merge_reward = info['merge_explored_reward']
        self.merge_ratio = delta_reward_all_map.sum() / self.total_cell_size #(self.width * self.height)
        info['merge_explored_ratio'] = self.merge_ratio
        info['merge_ratio_step'] = self.merge_ratio_step
        for i in range(self.num_agents):
            info["agent{}_ratio_step".format(i)] = self.agent_ratio_step[i]

        dones = np.array([done for agent_id in range(self.num_agents)])
        if self.use_single_reward:
            rewards = 0.3 * np.expand_dims(info['agent_explored_reward'], axis=1) + 0.7 * np.expand_dims(np.array([info['merge_explored_reward'] for _ in range(self.num_agents)]), axis=1)
        else:
            rewards = np.expand_dims(np.array([info['merge_explored_reward'] for _ in range(self.num_agents)]), axis=1)

        obs = np.array(obs)

        self.plot_map_with_path()

        return obs, rewards, dones, info



    def update_state(self, action):
        for i in range(self.num_agents):
            self.agent_pos[i][0] += 1
            self.agent_pos[i][1] += 1

    def get_short_term_goal(self, data):
        map_goal = []
        for e in range(self.num_agents):
            goal = [int(self.width*data['global_goal'][e][0]), int(self.height*data['global_goal'][e][1])]
            self.visualize_goal[e] = goal
            occupancy_grid = data['global_obs'][e, 0] + data['global_obs'][e, 1]
            # obstacle: 2  unknown: 0   free: 1
            frs, _ = self.detect_frontiers(occupancy_grid)
            # cluster targets into different groups and find the center of each group.
            target_process = copy.deepcopy(frs)
            cluster_center = []
            infoGain_cluster = []
            # path = []
            # currentLoc = self.continuous_to_discrete(self.robot.pos)
            # path.append(currentLoc)
            while(len(target_process) > 0):
                target_cluster = []
                target_cluster.append(target_process.pop())

                condition = True
                while(condition):
                    condition = False
                    size_target_process = len(target_process)
                    for i in reversed(range(size_target_process)):
                        for j in range(len(target_cluster)):
                            dis = abs(target_process[i][0] - target_cluster[j][0]) +  abs(target_process[i][1] - target_cluster[j][1])
                            if dis < 3:
                                target_cluster.append(target_process[i])
                                del target_process[i]
                                condition = True
                                break

                center_ = [0, 0]
                num_ = len(target_cluster)
                for i in range(num_):
                    center_[0] += target_cluster[i][0]
                    center_[1] += target_cluster[i][1]

                center_float = [float(center_[0])/float(num_), float(center_[1])/float(num_)]
                min_dis_ = 100.0
                min_idx_ = 10000
                for i in range(num_):
                    temp_dis_ = abs(center_float[0]-float(target_cluster[i][0])) + abs(center_float[1]-float(target_cluster[i][1]))
                    if temp_dis_ < min_dis_:
                        min_dis_ = temp_dis_
                        min_idx_ = i

                cluster_center.append([target_cluster[min_idx_][0], target_cluster[min_idx_][1]])
                infoGain_cluster.append(num_)
            free_cluster_center = []
            for i in range(len(cluster_center)):
                # find the nearest free grid
                for x in range(3):
                    for y in range(3):
                        if self.built_map[e][cluster_center[i][0]-1+x, cluster_center[i][1]-1+y] == 254:
                            free_cluster_center.append([cluster_center[i][0]-1+x, cluster_center[i][1]-1+y])
                            break
                    else:
                        continue
                    break
            if len(free_cluster_center) == 0:
                map_goal.append(self.agent_pos[e])
                print("cannot detect valid frontiers")
            else:
            # choose the frontier which is closest to the goal
                min_dis = 10000
                min_idx = -1
                for idx, fr in enumerate(free_cluster_center):
                    dis = math.sqrt(math.hypot(fr[0]-goal[0], fr[1]-goal[1]))
                    if dis < min_dis:
                        min_dis = dis
                        min_idx = idx
                map_goal.append(free_cluster_center[min_idx])

        # if self.visualization:
        #     self.visualize_map = copy.deepcopy(self.complete_map)
        #     for pt in self.visualize_goal:
        #         if pt[0] > 0 and pt[0] < 299 and pt[1] > 0 and pt[1] < 299:
        #             self.visualize_map[pt[0], pt[1]] = 128
        #             self.visualize_map[pt[0]-1, pt[1]] = 128
        #             self.visualize_map[pt[0]+1, pt[1]] = 128
        #             self.visualize_map[pt[0], pt[1]-1] = 128
        #             self.visualize_map[pt[0]-1, pt[1]-1] = 128
        #             self.visualize_map[pt[0]+1, pt[1]-1] = 128
        #             self.visualize_map[pt[0], pt[1]+1] = 128
        #             self.visualize_map[pt[0]-1, pt[1]+1] = 128
        #             self.visualize_map[pt[0]+1, pt[1]+1] = 128
        #         else:
        #             self.visualize_map[pt[0], pt[1]] = 128

        #     self.window.show_img(self.visualize_map)
        return np.array(map_goal)

    def detect_frontiers(self, explored_map):
        '''
        detect frontiers from current built map
        '''
        obstacles = []
        frontiers = []
        height = explored_map.shape[0]
        width = explored_map.shape[1]
        for i in range(2, height-2):
            for j in range(2, width-2):
                if explored_map[i][j] == 2:
                    obstacles.append([i,j])
                elif explored_map[i][j] == 0:
                    numFree = 0
                    temp1 = 0
                    if explored_map[i+1][j] == 1:
                        temp1 += 1 if explored_map[i+2][j] == 1 else 0
                        temp1 += 1 if explored_map[i+1][j+1] == 1 else 0
                        temp1 += 1 if explored_map[i+1][j-1] == 1 else 0
                        numFree += (temp1 > 0)
                    if explored_map[i][j+1] == 1:
                        temp1 += 1 if explored_map[i][j+2] == 1 else 0
                        temp1 += 1 if explored_map[i+1][j+1] == 1 else 0
                        temp1 += 1 if explored_map[i-1][j+1] == 1 else 0
                        numFree += (temp1 > 0)     
                    if explored_map[i-1][j] == 1:
                        temp1 += 1 if explored_map[i-1][j+1] == 1 else 0
                        temp1 += 1 if explored_map[i-1][j-1] == 1 else 0
                        temp1 += 1 if explored_map[i-2][j] == 1 else 0
                        numFree += (temp1 > 0)
                    if explored_map[i][j-1] == 1:
                        temp1 += 1 if explored_map[i][j-2] == 1 else 0
                        temp1 += 1 if explored_map[i+1][j-1] == 1 else 0
                        temp1 += 1 if explored_map[i-1][j-1] == 1 else 0
                        numFree += (temp1 > 0)     
                    if numFree > 0:
                        frontiers.append([i,j])      
        return frontiers, obstacles

    def visualize_path(self):
        vis_map = copy.deepcopy(self.built_map)
        for pos in self.grid_traj:
            vis_map[pos[0], pos[1]] = 64
        return vis_map

    def build_map_per_frame(self, pos, orn, gt_map, resolution, sensor_range):
        '''
        build map from a specific pos with a omni-directional scan
        pos: [x, y]
        '''
        grid_range = int(sensor_range / resolution)
        # width = 6*grid_range
        # height = 6*grid_range
        height = gt_map.shape[0]
        width = gt_map.shape[1]
        init_map = np.zeros((width, height))
        x, y = int(height/2+pos[0]/resolution-grid_range), int(width/2+pos[1]/resolution-grid_range) # relative to the upper left corner of the picture
        init_map = gt_map[x:x+2*grid_range, y:y+2*grid_range]
        # debug: show the robot's pos
        # gt_map[int(height/2+pos[0]), int(width/2+pos[1])] = 0
        # img = Image.fromarray(gt_map.astype('uint8'))
        # img.save("debug.pgm")
        # img.show()
        # import pdb; pdb.set_trace()
        # put mask
        mask_map = copy.deepcopy(init_map)
        mask_map_origin = [mask_map.shape[0]/2, mask_map.shape[1]/2]
        # debug: show the robot's pos
        # mask_map[int(mask_map.shape[0]/2), int(mask_map.shape[1]/2)] = 0
        # img = Image.fromarray(mask_map.astype('uint8'))
        # img.save("debug_mask.pgm")
        # img.show()
        # import pdb; pdb.set_trace()
        for i in range(mask_map.shape[0]):
            for j in range(mask_map.shape[1]):
                dist = math.sqrt((i+0.5-mask_map_origin[0])*(i+0.5-mask_map_origin[0])+(j+0.5-mask_map_origin[1])*(j+0.5-mask_map_origin[1]))
                # unknown
                if dist > grid_range:
                    mask_map[i][j] = 205
                else:
                    # print(j)
                    # left upper
                    flag = True
                    if i + 0.5 - mask_map.shape[0]/2 < -1 and j + 0.5 - mask_map.shape[1]/2 < -1:
                        flag = check_visible(i+1, j+1, init_map)
                    # left down
                    elif i + 0.5 - mask_map.shape[0]/2 > 1 and j + 0.5 - mask_map.shape[1]/2 < -1:
                        flag = check_visible(i, j+1, init_map)
                    # right upper
                    elif i + 0.5 - mask_map.shape[0]/2 < -1 and j + 0.5 - mask_map.shape[1]/2 > 1:
                        flag = check_visible(i+1, j, init_map)
                    # right down
                    elif i + 0.5 - mask_map.shape[0]/2 > 1 and j + 0.5 - mask_map.shape[1]/2 > 1:
                        flag = check_visible(i, j, init_map)
                    elif i - mask_map.shape[0]/2 == -1 and j - mask_map.shape[1]/2 < -1:
                        flag = check_visible(i, j+1, init_map)
                    elif i - mask_map.shape[0]/2 == -1 and j - mask_map.shape[1]/2 > 1:
                        flag = check_visible(i, j, init_map)
                    elif i - mask_map.shape[0]/2 == 0 and j - mask_map.shape[1]/2 < -1:
                        flag = check_visible(i+1, j+1, init_map)
                    elif i - mask_map.shape[0]/2 == 0 and j - mask_map.shape[1]/2 > 1:
                        flag = check_visible(i+1, j, init_map)
                    elif i - mask_map.shape[0]/2 < -1 and j - mask_map.shape[1]/2 == -1:
                        flag = check_visible(i+1, j, init_map)
                    elif i - mask_map.shape[0]/2 > 1 and j - mask_map.shape[1]/2 == -1:
                        flag = check_visible(i, j, init_map)
                    elif i - mask_map.shape[0]/2 < -1 and j - mask_map.shape[1]/2 == 0:
                        flag = check_visible(i+1, j+1, init_map)
                    elif i - mask_map.shape[0]/2 > 1 and j - mask_map.shape[1]/2 == 0:
                        flag = check_visible(i, j+1, init_map)
                    if not flag:
                        mask_map[i][j] = 205
        scale_map = np.zeros(gt_map.shape)
        scale_map[:,:] = 205
        scale_map[x:x+2*grid_range, y:y+2*grid_range] = mask_map
        return mask_map, scale_map

    def optimized_build_map(self, pos, orn, gt_map, resolution, sensor_range):
        '''
        build map from a specific pos with a omni-directional scan
        pos: [x, y]
        '''
        grid_range = int(sensor_range / resolution)
        height = gt_map.shape[0]
        width = gt_map.shape[1]
        # init_map = np.zeros((width, height))
        x, y = int(height/2+pos[0]/resolution), int(width/2+pos[1]/resolution)
        x_min, y_min = max(0, x-grid_range), max(0, y-grid_range)
        x_max, y_max = min(x+grid_range, height), min(y+grid_range, width)
        # x, y = int(height/2+pos[0]/resolution-grid_range), int(width/2+pos[1]/resolution-grid_range) # relative to the upper left corner of the picture
        init_map = gt_map[x_min:x_max, y_min:y_max]
        mask_map = copy.deepcopy(init_map)
        mask_map_origin = [x-x_min, y-y_min]

        for j in range(mask_map.shape[1]):
            laser_path = self.optimized_simulate_laser(0, j, init_map, mask_map_origin)
            path_map = self.plot_path(laser_path, mask_map)
            laser_path.reverse()
            laser_path.append([0,j])
            for idx, p in enumerate(laser_path[:-1]):
                if (init_map[p[0],p[1]] == 0 and init_map[laser_path[idx+1][0], laser_path[idx+1][1]] > 0) or (init_map[p[0],p[1]] == 0 and init_map[laser_path[idx+1][0], laser_path[idx+1][1]] == 0 and p[1] != laser_path[idx+1][1]) or (init_map[p[0],p[1]] == 0 and p[1] == mask_map_origin[1] - 1) or (init_map[p[0],p[1]] == 0 and p[1] == mask_map_origin[1]):
                    for pp in laser_path[idx+1:]:
                        mask_map[pp[0],pp[1]] = 205
                    break     
        for j in range(mask_map.shape[1]):
            laser_path = self.optimized_simulate_laser(mask_map.shape[0]-1, j, init_map, mask_map_origin)
            laser_path.reverse()
            laser_path.append([mask_map.shape[0]-1,j])
            for idx, p in enumerate(laser_path[:-1]):
                if (init_map[p[0],p[1]] == 0 and init_map[laser_path[idx+1][0], laser_path[idx+1][1]] > 0) or (init_map[p[0],p[1]] == 0 and init_map[laser_path[idx+1][0], laser_path[idx+1][1]] == 0 and p[1] != laser_path[idx+1][1]) or (init_map[p[0],p[1]] == 0 and p[1] == mask_map_origin[1] - 1) or (init_map[p[0],p[1]] == 0 and p[1] == mask_map_origin[1]):
                    for pp in laser_path[idx+1:]:
                        mask_map[pp[0],pp[1]] = 205
                    break  
        for i in range(mask_map.shape[0]):
            laser_path = self.optimized_simulate_laser(i, 0, init_map, mask_map_origin)
            laser_path.reverse()
            laser_path.append([i,0])
            for idx, p in enumerate(laser_path[:-1]):
                if (init_map[p[0],p[1]] == 0 and init_map[laser_path[idx+1][0], laser_path[idx+1][1]] > 0) or (init_map[p[0],p[1]] == 0 and init_map[laser_path[idx+1][0], laser_path[idx+1][1]] == 0 and p[0] != laser_path[idx+1][0]) or (init_map[p[0],p[1]] == 0 and p[0] == mask_map_origin[0] - 1) or (init_map[p[0],p[1]] == 0 and p[0] == mask_map_origin[0]):
                    for pp in laser_path[idx+1:]:
                        mask_map[pp[0],pp[1]] = 205
                    break      
        for i in range(mask_map.shape[0]):
            laser_path = self.optimized_simulate_laser(i, mask_map.shape[1]-1, init_map, mask_map_origin)
            laser_path.reverse()
            laser_path.append([i,mask_map.shape[1]-1])
            for idx, p in enumerate(laser_path[:-1]):
                if (init_map[p[0],p[1]] == 0 and init_map[laser_path[idx+1][0], laser_path[idx+1][1]] > 0) or (init_map[p[0],p[1]] == 0 and init_map[laser_path[idx+1][0], laser_path[idx+1][1]] == 0 and p[0] != laser_path[idx+1][0]) or (init_map[p[0],p[1]] == 0 and p[0] == mask_map_origin[0] - 1) or (init_map[p[0],p[1]] == 0 and p[0] == mask_map_origin[0]):
                    for pp in laser_path[idx+1:]:
                        mask_map[pp[0],pp[1]] = 205
                    break
        scale_map = np.zeros(gt_map.shape)
        scale_map[:,:] = 205
        scale_map[x_min:x_max, y_min:y_max] = mask_map
        for w in range(3):
            for h in range(3):
                scale_map[x-1+w, y-1+h] = gt_map[x-1+w, y-1+h]
        return mask_map, scale_map, (x_min, x_max), (y_min, y_max) 

    def build_map(self, pos, orn, gt_map, resolution, sensor_range):
        grid_range = int(sensor_range / resolution)
        height = gt_map.shape[0]
        width = gt_map.shape[1]
        init_map = np.zeros((width, height))
        x, y = int(height/2+pos[0]/resolution-grid_range), int(width/2+pos[1]/resolution-grid_range) # relative to the upper left corner of the picture
        init_map = gt_map[x:x+2*grid_range, y:y+2*grid_range]

        mask_map = copy.deepcopy(init_map)
        mask_map_origin = [mask_map.shape[0]/2, mask_map.shape[1]/2]

        for j in range(mask_map.shape[1]):
            laser_path = self.simulate_laser(0, j, mask_map)
            laser_path.reverse()
            for idx, p in enumerate(laser_path):
                if mask_map[p[0],p[1]] == 0:
                    for pp in laser_path[idx+1:]:
                        mask_map[pp[0],pp[1]] = 205
                    break
        img = Image.fromarray(mask_map.astype('uint8'))
        img.show()
        import pdb; pdb.set_trace()
        for j in range(mask_map.shape[1]):
            laser_path = self.simulate_laser(mask_map.shape[0]-1, j, mask_map)
            laser_path.reverse()
            for idx, p in enumerate(laser_path):
                if mask_map[p[0],p[1]] == 0:
                    for pp in laser_path[idx+1:]:
                        mask_map[pp[0],pp[1]] = 205
                    break
        for i in range(mask_map.shape[0]):
            laser_path = self.simulate_laser(i, 0, mask_map)
            laser_path.reverse()
            for idx, p in enumerate(laser_path):
                if mask_map[p[0],p[1]] == 0:
                    for pp in laser_path[idx+1:]:
                        mask_map[pp[0],pp[1]] = 205
                    break
        for i in range(mask_map.shape[0]):
            laser_path = self.simulate_laser(i, mask_map.shape[1]-1, mask_map)
            laser_path.reverse()
            for idx, p in enumerate(laser_path):
                if mask_map[p[0],p[1]] == 0:
                    for pp in laser_path[idx+1:]:
                        mask_map[pp[0],pp[1]] = 205
                    break
        scale_map = np.zeros(gt_map.shape)
        scale_map[:,:] = 205
        scale_map[x:x+2*grid_range, y:y+2*grid_range] = mask_map
        return mask_map, scale_map

    def simulate_laser(self, i, j, map):
        '''
        return a idx laser path
        '''
        laser_path = []
        if i < map.shape[0]/2 and j < map.shape[1]/2:
            step_length = map.shape[0]/2
            path_x = np.linspace(i, map.shape[0]/2-1, int(step_length))
            path_y = np.linspace(j, map.shape[1]/2-1, int(step_length))
            for i in range(int(step_length)):
                # print(int(math.floor(path_x[i])))
                # print(int(math.floor(path_y[i])))
                laser_path.append([int(math.floor(path_x[i])), int(math.floor(path_y[i]))])
        if i >= map.shape[0]/2 and j < map.shape[1]/2:
            step_length = map.shape[0]/2
            path_x = np.linspace(i, map.shape[0]/2, int(step_length))
            path_y = np.linspace(j, map.shape[1]/2-1, int(step_length))
            for i in range(int(step_length)):
                # print(int(math.floor(path_x[i])))
                # print(int(math.floor(path_y[i])))
                laser_path.append([int(math.ceil(path_x[i])), int(math.floor(path_y[i]))])
        if i < map.shape[0]/2 and j >= map.shape[1]/2:
            step_length = map.shape[0]/2
            path_x = np.linspace(i, map.shape[0]/2-1, int(step_length))
            path_y = np.linspace(j, map.shape[1]/2, int(step_length))
            for i in range(int(step_length)):
                # print(int(math.floor(path_x[i])))
                # print(int(math.floor(path_y[i])))
                laser_path.append([int(math.floor(path_x[i])), int(math.ceil(path_y[i]))])
        if i >= map.shape[0]/2 and j >= map.shape[1]/2:
            step_length = map.shape[0]/2
            path_x = np.linspace(i, map.shape[0]/2, int(step_length))
            path_y = np.linspace(j, map.shape[1]/2, int(step_length))
            for i in range(int(step_length)):
                # print(int(math.floor(path_x[i])))
                # print(int(math.floor(path_y[i])))
                laser_path.append([int(math.ceil(path_x[i])), int(math.ceil(path_y[i]))])
        return laser_path

    def optimized_simulate_laser(self, i, j, mask_map, map_origin):
        # left upper
        if i + 0.5 - map_origin[0] < -1 and j + 0.5 - map_origin[1] < -1:
            path = self.return_laser_path(i+1, j+1, mask_map, map_origin)
        # left down
        elif i + 0.5 - map_origin[0] > 1 and j + 0.5 - map_origin[1] < -1:
            path = self.return_laser_path(i, j+1, mask_map, map_origin)
        # right upper
        elif i + 0.5 - map_origin[0] < -1 and j + 0.5 - map_origin[1] > 1:
            path = self.return_laser_path(i+1, j, mask_map, map_origin)
        # right down
        elif i + 0.5 - map_origin[0] > 1 and j + 0.5 - map_origin[1] > 1:
            path = self.return_laser_path(i, j, mask_map, map_origin)
        elif i - map_origin[0] == -1 and j - map_origin[1] < -1:
            path = self.return_laser_path(i, j+1, mask_map, map_origin)
        elif i - map_origin[0] == -1 and j - map_origin[1] > 1:
            path = self.return_laser_path(i, j, mask_map, map_origin)
        elif i - map_origin[0] == 0 and j - map_origin[1] < -1:
            path = self.return_laser_path(i+1, j+1, mask_map, map_origin)
        elif i - map_origin[0] == 0 and j - map_origin[1] > 1:
            path = self.return_laser_path(i+1, j, mask_map, map_origin)
        elif i - map_origin[0] < -1 and j - map_origin[1] == -1:
            path = self.return_laser_path(i+1, j, mask_map, map_origin)
        elif i - map_origin[0] > 1 and j - map_origin[1] == -1:
            path = self.return_laser_path(i, j, mask_map, map_origin)
        elif i - map_origin[0] < -1 and j - map_origin[1] == 0:
            path = self.return_laser_path(i+1, j+1, mask_map, map_origin)
        elif i - map_origin[0] > 1 and j - map_origin[1] == 0:
            path = self.return_laser_path(i, j+1, mask_map, map_origin)
        return path
    
    def return_laser_path(self, i, j, map, map_origin):
        '''
        check whether [i, j] pixel is visible from the origin of map
        '''
        laser_path = []
        step_length = max(abs(i - map_origin[0]),
                        abs(j - map_origin[1]))
        path_x = np.linspace(i, map_origin[0], int(step_length)+2)
        path_y = np.linspace(j, map_origin[1], int(step_length)+2)
        for i in range(1, int(step_length)+1):
            # print(int(math.floor(path_x[i])))
            # print(int(math.floor(path_y[i])))
            laser_path.append([int(math.floor(path_x[i])), int(math.floor(path_y[i]))])
        return laser_path

    def build_map_given_path(self, path):
        for pose in path:
            _, map_this_frame, (x_min, x_max), (y_min, y_max) = self.optimized_build_map(self.discrete_to_continuous(pose[0]), 0, self.gt_map, self.resolution, self.sensor_range)  # can be modified to replace self.gt_map, self.resolution and self.sensor_range
            self.built_map = self.merge_two_map(self.built_map, map_this_frame, [x_min, x_max], [y_min, y_max])
        self.inflation_built_map = obstacle_inflation(self.built_map, 0.15, 0.05)

    def build_map_given_path_for_multi_robot(self, path, agent_id):
        # for pose in path:
        #     _, map_this_frame, (x_min, x_max), (y_min, y_max) = self.optimized_build_map(self.discrete_to_continuous(pose[0]), 0, self.gt_map, self.resolution, self.sensor_range)  # can be modified to replace self.gt_map, self.resolution and self.sensor_range
        #     self.built_map[agent_id] = self.merge_two_map(self.built_map[agent_id], map_this_frame, [x_min, x_max], [y_min, y_max])
        for i in range(int(len(path)/3)):
            _, map_this_frame, (x_min, x_max), (y_min, y_max) = self.optimized_build_map(self.discrete_to_continuous(path[i*3+1][0]), 0, self.gt_map, self.resolution, self.sensor_range)  # can be modified to replace self.gt_map, self.resolution and self.sensor_range
            self.built_map[agent_id] = self.merge_two_map(self.built_map[agent_id], map_this_frame, [x_min, x_max], [y_min, y_max])
        _, map_this_frame, (x_min, x_max), (y_min, y_max) = self.optimized_build_map(self.discrete_to_continuous(path[-1][0]), 0, self.gt_map, self.resolution, self.sensor_range)  # can be modified to replace self.gt_map, self.resolution and self.sensor_range
        self.built_map[agent_id] = self.merge_two_map(self.built_map[agent_id], map_this_frame, [x_min, x_max], [y_min, y_max])
        # self.inflation_built_map = obstacle_inflation(self.built_map, 0.15, 0.05)

    def merge_two_map(self, map1, map2, x, y):
        '''
        merge two map into one map
        should be accelerated
        '''
        # merge_map = map1 + map2
        # for i in range(merge_map.shape[0]):
        #     for j in range(merge_map.shape[1]):
        #         if merge_map[i][j] == 0 or merge_map[i][j] == 205 or merge_map[i][j] == 254:
        #             merge_map[i][j] = 0
        #         elif merge_map[i][j] == 410:
        #             merge_map[i][j] = 205
        #         elif merge_map[i][j] == 459 or merge_map[i][j] == 508:
        #             merge_map[i][j] = 254
        test_map = map1 + map2
        merge_map = copy.deepcopy(map1)
        for i in range(x[0], x[1]):
            for j in range(y[0], y[1]):
                if test_map[i][j] == 0 or test_map[i][j] == 205 or test_map[i][j] == 254:
                    merge_map[i][j] = 0
                elif test_map[i][j] == 410:
                    merge_map[i][j] = 205
                elif test_map[i][j] == 459 or test_map[i][j] == 508:
                    merge_map[i][j] = 254
        return merge_map

    def update_pose(self, c_pos, c_orn, linear_velocity, angular_velocity, step_time):
        if abs(angular_velocity) < 1e-3 :
            r = linear_velocity * step_time
            delta_x = r*math.sin(c_orn)
            delta_y = r*math.cos(c_orn)
            delta_theta = 0
            update_x = c_pos[0] - delta_x
            update_y = c_pos[1] + delta_y
            update_orn = c_orn + delta_theta    
        else:
            r = linear_velocity / angular_velocity
            delta_theta = angular_velocity * step_time
            delta_x = r*math.cos(c_orn) - r*math.cos(c_orn+delta_theta)
            delta_y = r*math.sin(c_orn+delta_theta)-r*math.sin(c_orn)
            update_x = c_pos[0] - delta_x
            update_y = c_pos[1] + delta_y
            update_orn = c_orn + delta_theta
        if update_orn > 2*math.pi:
            update_orn = update_orn - 2*math.pi
        if update_orn < 0:
            update_orn = update_orn + 2*math.pi
        return [update_x, update_y], update_orn

    def update_motion(self, c_pos, c_orn, linear_velocity, angular_velocity):
        '''
        return a list of 
        '''
        pos = c_pos
        orn = c_orn
        init_grid_idx = self.continuous_to_discrete(pos)
        ratio = math.ceil((abs(linear_velocity) * self.step_time) / self.resolution) + 1
        grid_experience = []
        pos_experience = []
        map_nodes = []
        for i in range(ratio):
            pos, orn = self.update_pose(pos, orn, linear_velocity, angular_velocity, self.step_time/ratio)
            grid_idx = self.continuous_to_discrete(pos)
            pos_experience.append(pos)
            grid_experience.append(grid_idx)
            if len(map_nodes) == 0:
                if grid_idx != init_grid_idx:
                    map_nodes.append(grid_idx)
            elif grid_idx != map_nodes[-1]:
                map_nodes.append(grid_idx)
        # self.robot.pos = pos
        # self.robot.orn = orn
        return pos, orn, pos_experience, grid_experience, map_nodes

    def calculate_traj(self, c_pos, c_orn, linear_velocity, angular_velocity, t):
        '''
        modify from update_motion (replace self.step_time with self.dw_time)
        '''
        pos = c_pos
        orn = c_orn
        init_grid_idx = self.continuous_to_discrete(pos)
        ratio = math.ceil((abs(linear_velocity) * t) / self.resolution) + 1
        grid_experience = []
        pos_experience = []
        map_nodes = []
        for i in range(ratio):
            pos, orn = self.update_pose(pos, orn, linear_velocity, angular_velocity, t/ratio)
            grid_idx = self.continuous_to_discrete(pos)
            pos_experience.append(pos)
            grid_experience.append(grid_idx)
            if len(map_nodes) == 0:
                if grid_idx != init_grid_idx:
                    map_nodes.append(grid_idx)
            elif grid_idx != map_nodes[-1]:
                map_nodes.append(grid_idx)
        # self.robot.pos = pos
        # self.robot.orn = orn
        return pos, orn, pos_experience, grid_experience, map_nodes

    def continuous_to_discrete(self, pos):
        idx_x = int(pos[0] / self.resolution) + int(self.gt_map.shape[0]/2)
        idx_y = int(pos[1] / self.resolution) + int(self.gt_map.shape[1]/2)
        return [idx_x, idx_y]

    def discrete_to_continuous(self, grid_idx):
        pos_x = (grid_idx[0] - self.gt_map.shape[0]/2) * self.resolution
        pos_y = (grid_idx[1] - self.gt_map.shape[1]/2) * self.resolution
        return [pos_x, pos_y]

    def dwa_planner(self, pos, orn, global_plan):
        """
        Calculates the optimal control input and trajectory from a given robot 
        state. This function performs the calculation of the dynamic window, the
        trajectory generation and the cost function calculation.

        It traverse all the possible trajectories and then return the optimal one.

        Keyword arguments:
        x -- numpy array containing the current robot state -> [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        """
        # First turn towards the goal
        d_pos = self.continuous_to_discrete(pos)
        dx = d_pos[0] - global_plan[-1][0]
        dy = d_pos[1] - global_plan[-1][1]
        # import pdb; pdb.set_trace()
        if dx >= 0 and dy < 0:
            if self.robot.orn >= 5*math.pi/4:
                return [0, self.robot.max_w], self.robot.pos
            elif self.robot.orn > math.pi/2 and self.robot.orn < 5*math.pi/4:
                return [0, self.robot.min_w], self.robot.pos
            else:
                pass
        if dx > 0 and dy >= 0:
            if self.robot.orn < math.pi/2:
                return [0, self.robot.max_w], self.robot.pos
            elif self.robot.orn > math.pi:
                return [0, self.robot.min_w], self.robot.pos
            else:
                pass  
        if dx <= 0 and dy > 0:
            if self.robot.orn < math.pi:
                return [0, self.robot.max_w], self.robot.pos
            elif self.robot.orn > 3*math.pi/2:
                return [0, self.robot.min_w], self.robot.pos
            else:
                pass  
        if dx < 0 and dy <= 0:
            if self.robot.orn < 3*math.pi/2 and self.robot.orn >= 3*math.pi/4:
                return [0, self.robot.max_w], self.robot.pos
            elif self.robot.orn < 3*math.pi/4:
                return [0, self.robot.min_w], self.robot.pos
            else:
                pass


        # Calculate the admissible velocities (DW) - [V_min, V_max, W_min, W_max] 
        dw_v = self.calculate_dw()

        # Calculate distance between current pos and goal
        goal = self.discrete_to_continuous(global_plan[-1])
        dis = math.hypot(self.robot.pos[0]-goal[0], self.robot.pos[1]-goal[1])
        # try to deal with the situation when the robot is close to frontier
        shortest_time = dis / self.robot.max_v
        plan_time=1
        # if shortest_time > 2:
        #     plan_time = 1
        # else:
        #     plan_time = shortest_time/3

        # Array of velocities (Initial, Final, Resolution)
        v_vect = np.arange(dw_v[0], dw_v[1], self.robot.v_res)
        w_vect = np.arange(dw_v[2], dw_v[3], self.robot.w_res)

        minimum_cost = float("inf")

        optimal_u = [0.0, 0.0]
        optimal_traj = pos

        cost_list = []
        idx_list = []
        u_list = []
        # Iterate through the linear velocties
        for v in v_vect:
            # Iterate through the angular velocties
            for w in w_vect:
                # Trajectory generation for the following three seconds
                _, _, pos_experience, grid_experience, map_nodes = self.calculate_traj(pos, orn, v, w, plan_time)
                # if len(map_nodes) == 0:
                #     continue
                if self.ObstacleCostFunction(map_nodes):
                    continue
                else:
                    MapGridCost, min_idx = self.point_to_path_min_distance(pos_experience[-1], global_plan)
                    cost_list.append(MapGridCost)
                    idx_list.append(min_idx)
                    u_list.append([v,w])
        
        idx_sorted_id = sorted(range(len(idx_list)), key=lambda k: idx_list[k], reverse=True)

        af_cost_list = []
        af_u_list = []
        if len(idx_sorted_id) >= 10:
            for i in range(10):
                af_cost_list.append(cost_list[idx_sorted_id[i]])
                af_u_list.append(u_list[idx_sorted_id[i]])
        else:
            for i in range(len(idx_sorted_id)):
                af_cost_list.append(cost_list[idx_sorted_id[i]])
                af_u_list.append(u_list[idx_sorted_id[i]])
        optimal_cost = min(af_cost_list)
        optimal_idx = af_cost_list.index(optimal_cost)
        optimal_u = af_u_list[optimal_idx]
        # print("#########")
        # import pdb; pdb.set_trace()
        return optimal_u, optimal_traj

    def naive_local_planner(self, global_plan):
        '''
        Naive local planner
        always move along the global path
        '''
        pose = []
        # add orn to global path
        for idx, pos in enumerate(global_plan):
            if idx == 0:
                pose.append(self.calculate_pose(c_pos=pos, n_pos=global_plan[idx+1]))
            elif idx == len(global_plan)-1:
                pose.append(self.calculate_pose(c_pos=pos, p_pos=global_plan[idx-1]))
            else:
                pose.append(self.calculate_pose(c_pos=pos, p_pos=global_plan[idx-1], n_pos=global_plan[idx+1]))
        return pose

    def calculate_pose(self, p_pos=None, c_pos=None, n_pos=None):
        '''
        For naive local planner only
        p_pos: previous robot's position
        c_pos: current robot's position
        n_pos: next robot's position
        '''
        # n_pos - c_pos
        start_pos2orn = {(-1,-1):3*math.pi/4, (-1,0):math.pi/2, (-1,1):math.pi/4, (0,1):0, (1,1):7*math.pi/4, (1,0):3*math.pi/2, (1,-1):5*math.pi/4, (0,-1):math.pi}
        if not p_pos:
            return [c_pos, start_pos2orn[tuple((np.array(n_pos)-np.array(c_pos)).tolist())]]
        # p_pos - c_pos
        end_pos2orn = {(-1,-1):7*math.pi/4, (-1,0):3*math.pi/2, (-1,1):5*math.pi/4, (0,1):math.pi, (1,1):3*math.pi/4, (1,0):math.pi/2, (1,-1):math.pi/4, (0,-1):0}
        if not n_pos:
            return [c_pos, end_pos2orn[tuple((np.array(p_pos)-np.array(c_pos)).tolist())]]

        # tuple (p_pos - c_pos, n_pos - c_pos)   
        mid_end_pos2orn = {(-1,-1,-1,1):0, (-1,-1,0,1):15*math.pi/8, (-1,-1,1,1):7*math.pi/4, (-1,-1,1,0):13*math.pi/8,(-1,-1,1,-1):3*math.pi/2,
                           (-1,0,0,1):7*math.pi/4, (-1,0,1,1):13*math.pi/8, (-1,0,1,0):3*math.pi/2, (-1,0,1,-1):11*math.pi/8, (-1,0,0,-1):5*math.pi/4,
                           (-1,1,1,1):3*math.pi/2, (-1,1,1,0):11*math.pi/8, (-1,1,1,-1):5*math.pi/4, (-1,1,0,-1):9*math.pi/8, (-1,1,-1,-1):math.pi,
                           (0,1,1,0):5*math.pi/4, (0,1,1,-1):9*math.pi/8, (0,1,0,-1):math.pi, (0,1,-1,-1):7*math.pi/8, (0,1,-1,0):3*math.pi/4,
                           (1,1,1,-1):math.pi, (1,1,0,-1):7*math.pi/8, (1,1,-1,-1):3*math.pi/4, (1,1,-1,0):5*math.pi/8, (1,1,-1,1):math.pi/2,
                           (1,0,0,-1):3*math.pi/4, (1,0,-1,-1):5*math.pi/8, (1,0,-1,0):math.pi/2, (1,0,-1,1):3*math.pi/8, (1,0,0,1):math.pi/4,
                           (1,-1,-1,-1):math.pi/2, (1,-1,-1,0):3*math.pi/8, (1,-1,-1,1):math.pi/4, (1,-1,0,1):math.pi/8, (1,-1,1,1):0,
                           (0,-1,-1,0):math.pi/4, (0,-1,-1,1):math.pi/8, (0,-1,0,1):0, (0,-1,1,1):15*math.pi/8, (0,-1,1,0):7*math.pi/4}
        return [c_pos, mid_end_pos2orn[tuple(np.concatenate([np.array(p_pos)-np.array(c_pos),np.array(n_pos)-np.array(c_pos)]).tolist())]]

    def calculate_dw(self):
        """
        Calculate the dynamic window, equivalent to the velocities that can be
        reached within the next time interval (dt). Accerelations are considered
        in this calculation as they are the hardware limitation.  
        
        Note that the dynamic window is centred around the actual velocity and 
        the extensions of it depend on the accelerations that can be exerted. All 
        curvatures outside the dynamic window cannot be reached within the next 
        time interval and thus are not considered for the obstacle avoidance.

        Keyword arguments:
        x -- numpy array containing the current robot state -> [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        u -- numpy array containing the control commands -> [v(m/s), omega(rad/s)]
        """
        # We need to make sure we are within the motion constraints, in other words we need to 
        # lie within the dynamic window, for that reason we consider the velocity boundaries
        dw = [
            self.robot.min_v, # Min lin. vel 
            self.robot.max_v, # Max lin. vel
            self.robot.min_w, # Min ang. vel
            self.robot.max_w  # Max ang. vel
        ]

        # Dynamic window -> [V_min, V_max, W_min, W_max]  
        return dw

    def plot_path(self, path, map):
        vis_map = copy.deepcopy(map)
        for pixel in path:
            vis_map[pixel[0],pixel[1]] = 128
        return vis_map

    def ObstacleCostFunction(self, trajectory):
        for each in trajectory:
            if self.inflation_map[each[0],each[1]] == 0:
                return True
            else:
                pass
        return False
    
    def MapGridCostFunction(self, trajectory, global_plan):
        pass

    def Astar_global_planner(self, start, goal):
        # start_pos = self.continuous_to_discrete(start)
        # goal_pos = self.continuous_to_discrete(goal)
        astar = AStar(tuple(start), tuple(goal), self.gt_map, "euclidean")
        # plot = plotting.Plotting(s_start, s_goal)
        path, visited = astar.searching()
        # vis_map = self.plot_path(path)
        # img = Image.fromarray(vis_map.astype('uint8'))
        # img.show()
        # import pdb; pdb.set_trace()
        return list(reversed(path))

    def point_to_path_min_distance(self, point, path):
        dis = []
        for each in path:
            d_each = self.discrete_to_continuous(each)
            dis.append(math.hypot(point[0]-d_each[0], point[1]-d_each[1]))
        return min(dis), dis.index(min(dis))

    def frontiers_detection(self):
        '''
        detect frontiers from current built map
        '''
        obstacles = []
        frontiers = []
        height = self.built_map.shape[0]
        width = self.built_map.shape[1]
        for i in range(2, height-2):
            for j in range(2, width-2):
                if self.built_map[i][j] == 0:
                    obstacles.append([i,j])
                elif self.built_map[i][j] == 205:
                    numFree = 0
                    temp1 = 0
                    if self.built_map[i+1][j] == 254:
                        temp1 += 1 if self.built_map[i+2][j] == 254 else 0
                        temp1 += 1 if self.built_map[i+1][j+1] == 254 else 0
                        temp1 += 1 if self.built_map[i+1][j-1] == 254 else 0
                        numFree += (temp1 > 0)
                    if self.built_map[i][j+1] == 254:
                        temp1 += 1 if self.built_map[i][j+2] == 254 else 0
                        temp1 += 1 if self.built_map[i+1][j+1] == 254 else 0
                        temp1 += 1 if self.built_map[i-1][j+1] == 254 else 0
                        numFree += (temp1 > 0)     
                    if self.built_map[i-1][j] == 254:
                        temp1 += 1 if self.built_map[i-1][j+1] == 254 else 0
                        temp1 += 1 if self.built_map[i-1][j-1] == 254 else 0
                        temp1 += 1 if self.built_map[i-2][j] == 254 else 0
                        numFree += (temp1 > 0)
                    if self.built_map[i][j-1] == 254:
                        temp1 += 1 if self.built_map[i][j-2] == 254 else 0
                        temp1 += 1 if self.built_map[i+1][j-1] == 254 else 0
                        temp1 += 1 if self.built_map[i-1][j-1] == 254 else 0
                        numFree += (temp1 > 0)     
                    if numFree > 0:
                        frontiers.append([i,j])      
        return frontiers, obstacles

    def visualize_frontiers(self):
        frs, _ = self.frontiers_detection()
        vis_map = copy.deepcopy(self.built_map)
        for fr in frs:
            vis_map[fr[0], fr[1]] = 128
        return vis_map

    def arrival_detection(self, goal):
        '''
        detect whether robot arrives goal
        goal: discrete idx 
        '''
        goal = self.discrete_to_continuous(goal)
        dis = math.hypot(self.robot.pos[0]-goal[0], self.robot.pos[1]-goal[1])
        if dis <= self.minDis2Frontier:
            return True
        else:
            return False

    def rrt_explore(self, eta=0.5):
        # set the origin of the RRT
        V = []
        xnew = [self.robot.pos[0], self.robot.pos[1]]
        V.append(xnew)

        detectFrontierNum = 0
        iteration_num = 0
        info_radius = 1.0
        hysteresis_radius = 3.0
        hysteresis_gain = 2.0


        # initialize the previous goal point
        previous_goal = [xnew[0]+3*self.resolution, xnew[1]+3*self.resolution]

        while(True):
            x_rand = [random.random()*self.built_map.shape[0], random.random()*self.built_map.shape[1]]
            x_nearest = self.Nearest(V, x_rand)
            x_new = self.Steer(x_nearest, x_rand, eta)

            checking = self.ObstacleFree(x_nearest, x_new)

            if checking == 2:
                self.frontiers.append(x_new)
            elif checking == 0:
                V.append(x_new)
            
            iteration_num += 1

            # add previous goal
            if self.mapValue(self.built_map, previous_goal) == 205:
                self.frontiers.append(previous_goal)

            # remove old frontiers whose places have already been explored & the invalid frontiers.
            for i in reversed(range(len(self.frontiers))):
                if self.mapValue(self.built_map, self.frontiers[i]) != 205:
                    del self.frontiers[i]

            if len(self.frontiers) - detectFrontierNum > 25 or iteration_num > 500:
                iteration_num = 0
                # initializes current goal's information
                current_goal_score = -500.0
                current_goal_infoGain = -500.0
                current_goal_idx   = -1
                detectFrontierNum = len(self.frontiers)

                # display valid frontiers

                # find the goal which has the highest score
                for fr in self.frontiers:
                    infoGain = self.informationRectangleGain(self.built_map, fr, info_radius)
                    travel_distance = self.distance(self.robot.pos, fr)
                    if travel_distance <= hysteresis_radius:
                        infoGain*=hysteresis_gain
                    score = 3*infoGain - travel_distance
                
                    if score > current_goal_score:
                        current_goal_score    = score
                        current_goal_infoGain = infoGain
                        current_goal_idx      = i
                
                if current_goal_idx == -1:
                    continue

                robotGoal = self.frontiers[current_goal_idx]
                if abs(previous_goal[0]-robotGoal[0])+abs(previous_goal[1]-robotGoal[1]) < 3*self.resolution:
                    continue
                else:
                    previous_goal = robotGoal

        pass

    def rrt_explore_single_thread(self, window_path, window_rrt, eta=0.20):
        # set the origin of the RRT
        V = []
        xnew = [self.robot.pos[0], self.robot.pos[1]]
        V.append(xnew)

        detectFrontierNum = 0
        iteration_num = 0
        info_radius = 1.0
        hysteresis_radius = 3.0
        hysteresis_gain = 2.0

        path_log = []

        # initialize the previous goal point
        previous_goal = [xnew[0]+3*self.resolution, xnew[1]+3*self.resolution]

        while(True):
            # find a goal
            while(True):
                start = time.time()
                x_rand = [random.random()*self.built_map.shape[0], random.random()*self.built_map.shape[1]]
                x_rand = self.discrete_to_continuous(x_rand)
                x_nearest = self.Nearest(V, x_rand)
                x_new = self.Steer(x_nearest, x_rand, eta)

                checking = self.ObstacleFree(x_nearest, x_new)

                if checking == 2:
                    self.frontiers.append(x_new)
                elif checking == 0:
                    V.append(x_new)
                print("one iter costs: ", time.time()-start)
                print("V length: ", len(V))
                print("frontiers length: ", len(self.frontiers))
                print("detectFrontierNum: ", detectFrontierNum)
                iteration_num += 1
                v_map = self.visualize_rrt(V)
                window_rrt.show_img(v_map)
                # add previous goal
                if self.mapValue(self.inflation_built_map, previous_goal) == 205:
                    self.frontiers.append(previous_goal)

                # remove old frontiers whose places have already been explored & the invalid frontiers.
                # for i in reversed(range(len(self.frontiers))):
                #     if self.mapValue(self.inflation_built_map, self.frontiers[i]) != 205:
                #         del self.frontiers[i]
                                    
                if len(self.frontiers) - detectFrontierNum > 20 or iteration_num > 500:
                    iteration_num = 0
                    # initializes current goal's information
                    current_goal_score = -500.0
                    current_goal_infoGain = -500.0
                    current_goal_idx   = -1
                    detectFrontierNum = len(self.frontiers)

                    # display valid frontiers

                    # find the goal which has the highest score
                    for idx, fr in enumerate(self.frontiers):
                        infoGain = self.informationRectangleGain(self.built_map, fr, info_radius)
                        travel_distance = self.distance(self.robot.pos, fr)
                        if travel_distance <= hysteresis_radius:
                            infoGain*=hysteresis_gain
                        score = 3*infoGain - travel_distance
                    
                        if score > current_goal_score:
                            current_goal_score    = score
                            current_goal_infoGain = infoGain
                            current_goal_idx      = idx
                    
                    if current_goal_idx == -1:
                        continue

                    robotGoal = self.frontiers[current_goal_idx]
                    # previous_goal = robotGoal
                    # print("robot goal:", robotGoal)
                    # break
                    if abs(previous_goal[0]-robotGoal[0])+abs(previous_goal[1]-robotGoal[1]) < 3*self.resolution:
                        import pdb; pdb.set_trace()
                        continue
                    else:
                        previous_goal = robotGoal
                        print("robot goal:", robotGoal)
                        break
            
            # choose a goal which is free, guaranteeing that A* can work normally
            discrete_goal = self.continuous_to_discrete(robotGoal)
            if self.gt_map[discrete_goal[0],discrete_goal[1]] == 254:
                free_robotGoal = robotGoal
            else:
                for i in range(discrete_goal[0]-4, discrete_goal[0]+5):
                    for j in range(discrete_goal[1]-4, discrete_goal[1]+5):
                        if self.gt_map[i][j] == 254:
                            free_robotGoal = self.discrete_to_continuous([i,j])
                            break
            global_plan = self.Astar_global_planner(self.robot.pos, free_robotGoal)   
            pose = self.naive_local_planner(global_plan)
            path_log.extend(global_plan)
            # self.visualize_rrt(V)
            # self.visualize_rrt_frontiers(self.frontiers)
            self.build_map_given_path(pose)
            self.robot.pos = self.discrete_to_continuous(pose[-1][0])

            # remove old frontiers whose places have already been explored & the invalid frontiers.
            for i in reversed(range(len(self.frontiers))):
                if self.mapValue(self.inflation_built_map, self.frontiers[i]) != 205:
                    del self.frontiers[i]

            vis_map = self.plot_path(path_log, self.built_map)
            window_path.show_img(vis_map)
            # img = Image.fromarray(vis_map.astype('uint8'))
            # img.save("debug.pgm")
            # img.show()

    def visualize_rrt(self, V, show=False):
        vis_map = copy.deepcopy(self.built_map)
        for pt in V:
            temp_pt = self.continuous_to_discrete(pt)
            vis_map[temp_pt[0],temp_pt[1]] = 128
        if show:
            img = Image.fromarray(vis_map.astype('uint8'))
            img.save("rrt.pgm")
            img.show()
        return vis_map

    def visualize_rrt_frontiers(self, frs, show=False):
        vis_map = copy.deepcopy(self.built_map)
        for fr in frs:
            d_fr = self.continuous_to_discrete(fr)
            vis_map[d_fr[0], d_fr[1]] = 128
        if show:
            img = Image.fromarray(vis_map.astype('uint8'))
            img.save("rrt_frs.pgm")
            img.show()
        return vis_map

    def Nearest(self, V, x):
        min_d = self.distance(V[0], x)
        for i in range(len(V)):
            temp = self.distance(V[i], x)
            if temp <= min_d:
                min_d = temp
                min_idx = i
        return V[min_idx]

    def distance(self, x1, x2):
        return math.sqrt(math.hypot(x1[0]-x2[0],x1[1]-x2[1]))

    def Steer(self, x_nearest, x_rand, eta):
        x_new = []
        if (self.distance(x_nearest, x_rand) <= eta):
            x_new = x_rand
        else:
            m = (x_rand[1]-x_nearest[1])/(x_rand[0]-x_nearest[0])
            x_new.append((np.sign(x_rand[0]-x_nearest[0]))*(math.sqrt( (math.pow(eta,2)) / ((math.pow(m,2))+1) )   )+x_nearest[0])
            x_new.append(m*(x_new[0]-x_nearest[0])+x_nearest[1])   
        return x_new     

    def ObstacleFree(self, xnear, xnew):
        rez = self.resolution*0.2
        stepz = int(math.ceil(self.distance(xnew,xnear))/rez)
        xi = xnear
        # Free='0', Frontier='2', Obstacle='1'
        # map data:  0 occupied      205 unknown       254 free
        for i in range(stepz):
            xi = self.Steer(xi, xnew, rez)
            if self.mapValue(self.inflation_built_map, xi) == 0:
                return 1
            if self.mapValue(self.inflation_built_map, xi) == 205:
                x_new=xi
                return 2
        return 0

    def gridValue(self, xp):
        xp = self.continuous_to_discrete(xp)
        return self.built_map[xp[0],xp[1]]

    def mapValue(self, mapData, point):
        point = self.continuous_to_discrete(point)
        return mapData[point[0], point[1]]

    def informationRectangleGain(self, mapData, point, r):
        infoGainValue = 0
        r_region = int(r/self.resolution)
        point = self.continuous_to_discrete(point)
        # if point[0]+r_region < mapData.shape[0] and point[1]+r_region < mapData.shape[1]:
        #     for i in range(point[0]-r_region, point[0]+r_region+1):
        #         for j in range(point[1]-r_region, point[1]+r_region+1):
        #             if mapData[i][j] == 205:
        #                 infoGainValue += 1
        #             elif mapData[i][j] == 0:
        #                 infoGainValue -= 1
        # else:
        for i in range(point[0]-r_region, min(point[0]+r_region+1, mapData.shape[0])):
            for j in range(point[1]-r_region, min(point[1]+r_region+1, mapData.shape[1])):
                if mapData[i][j] == 205:
                    infoGainValue += 1
                elif mapData[i][j] == 0:
                    infoGainValue -= 1
        tempResult = infoGainValue*math.pow(self.resolution, 2)
        return tempResult

    # For APF exploration
    def mmpf_explore(self, window_path, window_cluster):
        path_log = []
        while(True):
            targets, obstacles = self.frontiers_detection()

            # cluster targets into different groups and find the center of each group.
            target_process = copy.deepcopy(targets)
            cluster_center = []
            infoGain_cluster = []

            path = []
            currentLoc = self.continuous_to_discrete(self.robot.pos)
            path.append(currentLoc)

            K_ATTRACT = 1
            riverFlowPotentialGain = 1

            while(len(target_process) > 0):
                target_cluster = []
                target_cluster.append(target_process.pop())

                condition = True
                while(condition):
                    condition = False
                    size_target_process = len(target_process)
                    for i in reversed(range(size_target_process)):
                        for j in range(len(target_cluster)):
                            dis = abs(target_process[i][0] - target_cluster[j][0]) +  abs(target_process[i][1] - target_cluster[j][1])
                            if dis < 3:
                                target_cluster.append(target_process[i])
                                del target_process[i]
                                condition = True
                                break

                center_ = [0, 0]
                num_ = len(target_cluster)
                for i in range(num_):
                    center_[0] += target_cluster[i][0]
                    center_[1] += target_cluster[i][1]

                center_float = [float(center_[0])/float(num_), float(center_[1])/float(num_)]
                min_dis_ = 100.0
                min_idx_ = 10000
                for i in range(num_):
                    temp_dis_ = abs(center_float[0]-float(target_cluster[i][0])) + abs(center_float[1]-float(target_cluster[i][1]))
                    if temp_dis_ < min_dis_:
                        min_dis_ = temp_dis_
                        min_idx_ = i

                cluster_center.append([target_cluster[min_idx_][0], target_cluster[min_idx_][1]])
                infoGain_cluster.append(num_)

            vis_map = self.visualize_mmpf_frontiers(cluster_center)
            window_cluster.show_img(vis_map)

            cluster_num = len(cluster_center)

            dismap_target = []

            for i in range(cluster_num):
                dismap_target.append(self.dismapConstruction_start_target(cluster_center[i]))

            currentLoc = self.continuous_to_discrete(self.robot.pos)
            # calculate path
            iteration = 1
            currentPotential = 10000
            riverFlowPotentialGain = 1
            minDis2Frontier  = 10000;  
            while (iteration < 3000 and minDis2Frontier > 1):
                potential = [0,0,0,0]
                min_idx = -1
                min_potential = 10000
                loc_around = [[currentLoc[0]-1, currentLoc[1]], # upper
                            [currentLoc[0], currentLoc[1]-1], # left
                            [currentLoc[0]+1, currentLoc[1]], # down
                            [currentLoc[0], currentLoc[1]+1]] # right
                for i in range(4):
                    curr_around = loc_around[i]
                    # calculate current potential
                    attract = 0
                    repulsive = 0
                    for j in range(len(cluster_center)):
                        temp = dismap_target[j][curr_around[0],curr_around[1]]
                        if temp < 1:
                            continue
                        attract = attract - K_ATTRACT*infoGain_cluster[j]/temp

                    # to increase the potential if currend point has been passed before
                    for j in range(len(path)):
                        if curr_around[0] == path[j][0] and curr_around[1] == path[j][1]:
                            attract += riverFlowPotentialGain*5

                    # Add impact of robots.

                    potential[i] = attract
                    if min_potential > potential[i]:
                        min_potential = potential[i]
                        min_idx = i

                if currentPotential > min_potential:
                    path.append(loc_around[min_idx])
                    currentPotential = min_potential
                else:
                    riverFlowPotentialGain += 1

                currentLoc = path[-1]

                for i in range(len(cluster_center)):
                    temp_dis_ = dismap_target[i][currentLoc[0],currentLoc[1]]
                    if temp_dis_ == 0 and abs(currentLoc[0]-cluster_center[i][0]) + abs(currentLoc[1]-cluster_center[i][1]) > 0:
                        continue
                    if minDis2Frontier > temp_dis_:
                        minDis2Frontier = temp_dis_
                iteration += 1

            robotGoal = self.discrete_to_continuous(path[-1])
            global_plan = self.Astar_global_planner(self.robot.pos, robotGoal)   
            pose = self.naive_local_planner(global_plan)
            path_log.extend(global_plan)
            # self.visualize_rrt(V)
            # self.visualize_rrt_frontiers(self.frontiers)
            self.build_map_given_path(pose)
            self.robot.pos = self.discrete_to_continuous(pose[-1][0])

            vis_map = self.plot_path(path_log, self.built_map)
            window_path.show_img(vis_map)
            import pdb; pdb.set_trace()

    def visualize_mmpf_frontiers(self, frs, show=False):
        vis_map = copy.deepcopy(self.built_map)
        for fr in frs:
            vis_map[fr[0], fr[1]] = 128
        if show:
            img = Image.fromarray(vis_map.astype('uint8'))
            img.save("mmpf_frs.pgm")
            img.show()
        return vis_map

    def dismapConstruction_start_target(self, curr, map):
        curr_iter = []
        next_iter = []

        iter = 1
        LARGEST_MAP_DISTANCE = 500*1000
        curr_iter.append(curr)

        dismap_backup = copy.deepcopy(map)
        dismap_ = copy.deepcopy(map)
        # dismap_: obstacle -2  unknown -1 free 0
        # built_map: obstacle 0 unknown 205 free 254
        for i in range(dismap_.shape[0]):
            for j in range(dismap_.shape[1]):
                if dismap_backup[i][j] == 0:
                    dismap_[i][j] = -2
                if dismap_backup[i][j] == 205:
                    dismap_[i][j] = -1
                if dismap_backup[i][j] == 254:
                    dismap_[i][j] = 0
        dismap_[curr[0], curr[1]] = -500

        while(len(curr_iter)) > 0:
            if iter > LARGEST_MAP_DISTANCE:
                print("distance exceeds MAXIMUM SETUP")
                return
            for i in range(len(curr_iter)):
                if dismap_[curr_iter[i][0]+1, curr_iter[i][1]] == 0:
                    dismap_[curr_iter[i][0]+1, curr_iter[i][1]] = iter
                    next_iter.append([curr_iter[i][0]+1, curr_iter[i][1]])
                if dismap_[curr_iter[i][0], curr_iter[i][1]+1] == 0:
                    dismap_[curr_iter[i][0], curr_iter[i][1]+1] = iter
                    next_iter.append([curr_iter[i][0], curr_iter[i][1]+1])           
                if dismap_[curr_iter[i][0]-1, curr_iter[i][1]] == 0:
                    dismap_[curr_iter[i][0]-1, curr_iter[i][1]] = iter
                    next_iter.append([curr_iter[i][0]-1, curr_iter[i][1]])
                if dismap_[curr_iter[i][0], curr_iter[i][1]-1] == 0:
                    dismap_[curr_iter[i][0], curr_iter[i][1]-1] = iter
                    next_iter.append([curr_iter[i][0], curr_iter[i][1]-1])  
            curr_iter = copy.deepcopy(next_iter)
            next_iter = []
            iter += 1

        dismap_[curr[0],curr[1]] = 0

        # window = Window('path')
        # window.show(block=False)
        # window.show_img(dismap_)
        # import pdb; pdb.set_trace()

        return dismap_    

    def plot_map_with_path(self):
        vis = copy.deepcopy(self.complete_map)
        for e in range(self.num_agents):
            for pose in self.path_log[e]:
                vis[pose[0], pose[1]] = 127
        self.window.show_img(vis)

    def frontiers_detection_for_cost(self, map):
        '''
        detect frontiers from current built map
        '''
        obstacles = []
        frontiers = []
        height = map.shape[0]
        width = map.shape[1]
        for i in range(2, height-2):
            for j in range(2, width-2):
                if map[i][j] == 0:
                    obstacles.append([i,j])
                elif map[i][j] == 205:
                    numFree = 0
                    temp1 = 0
                    if map[i+1][j] == 254:
                        temp1 += 1 if map[i+2][j] == 254 else 0
                        temp1 += 1 if map[i+1][j+1] == 254 else 0
                        temp1 += 1 if map[i+1][j-1] == 254 else 0
                        numFree += (temp1 > 0)
                    if map[i][j+1] == 254:
                        temp1 += 1 if map[i][j+2] == 254 else 0
                        temp1 += 1 if map[i+1][j+1] == 254 else 0
                        temp1 += 1 if map[i-1][j+1] == 254 else 0
                        numFree += (temp1 > 0)     
                    if map[i-1][j] == 254:
                        temp1 += 1 if map[i-1][j+1] == 254 else 0
                        temp1 += 1 if map[i-1][j-1] == 254 else 0
                        temp1 += 1 if map[i-2][j] == 254 else 0
                        numFree += (temp1 > 0)
                    if map[i][j-1] == 254:
                        temp1 += 1 if map[i][j-2] == 254 else 0
                        temp1 += 1 if map[i+1][j-1] == 254 else 0
                        temp1 += 1 if map[i-1][j-1] == 254 else 0
                        numFree += (temp1 > 0)     
                    if numFree > 0:
                        frontiers.append([i,j])      
        return frontiers, obstacles

    def get_goal_for_cost(self):
        map_goal = []
        for e in range(self.num_agents):
            # goal = [int(self.width*data['global_goal'][e][0]), int(self.height*data['global_goal'][e][1])]
            # self.visualize_goal[e] = goal
            # occupancy_grid = data['global_obs'][e, 0] + data['global_obs'][e, 1]
            # obstacle: 2  unknown: 0   free: 1
            frs, _ = self.frontiers_detection_for_cost(self.complete_map)
            # cluster targets into different groups and find the center of each group.
            target_process = copy.deepcopy(frs)
            cluster_center = []
            infoGain_cluster = []
            # path = []
            # currentLoc = self.continuous_to_discrete(self.robot.pos)
            # path.append(currentLoc)
            while(len(target_process) > 0):
                target_cluster = []
                target_cluster.append(target_process.pop())

                condition = True
                while(condition):
                    condition = False
                    size_target_process = len(target_process)
                    for i in reversed(range(size_target_process)):
                        for j in range(len(target_cluster)):
                            dis = abs(target_process[i][0] - target_cluster[j][0]) +  abs(target_process[i][1] - target_cluster[j][1])
                            if dis < 3:
                                target_cluster.append(target_process[i])
                                del target_process[i]
                                condition = True
                                break

                center_ = [0, 0]
                num_ = len(target_cluster)
                for i in range(num_):
                    center_[0] += target_cluster[i][0]
                    center_[1] += target_cluster[i][1]

                center_float = [float(center_[0])/float(num_), float(center_[1])/float(num_)]
                min_dis_ = 100.0
                min_idx_ = 10000
                for i in range(num_):
                    temp_dis_ = abs(center_float[0]-float(target_cluster[i][0])) + abs(center_float[1]-float(target_cluster[i][1]))
                    if temp_dis_ < min_dis_:
                        min_dis_ = temp_dis_
                        min_idx_ = i

                cluster_center.append([target_cluster[min_idx_][0], target_cluster[min_idx_][1]])
                infoGain_cluster.append(num_)
            # free_cluster_center = []
            # for i in range(len(cluster_center)):
            #     # find the nearest free grid
            #     for x in range(3):
            #         for y in range(3):
            #             if self.built_map[e][cluster_center[i][0]-1+x, cluster_center[i][1]-1+y] == 254:
            #                 free_cluster_center.append([cluster_center[i][0]-1+x, cluster_center[i][1]-1+y])
            #                 break
            #         else:
            #             continue
            #         break
            
            # curr_dismap = self.dismapConstruction_start_target(self.agent_pos[e], self.built_map[e])
            curr_dismap = self.dismapConstruction_start_target(self.agent_pos[e], self.complete_map)
            Dis2Frs = []
            free_cluster_center = []
            for i in range(len(cluster_center)):
                # find the nearest free grid
                for x in range(3):
                    for y in range(3):
                        # if self.built_map[e][cluster_center[i][0]-1+x, cluster_center[i][1]-1+y] == 254:
                        if self.complete_map[cluster_center[i][0]-1+x, cluster_center[i][1]-1+y] == 254:
                            Dis2Frs.append(curr_dismap[cluster_center[i][0]-1+x, cluster_center[i][1]-1+y])
                            free_cluster_center.append([cluster_center[i][0]-1+x, cluster_center[i][1]-1+y])
                            break
                    else:
                        continue
                    break
            
            map_goal.append(free_cluster_center[Dis2Frs.index(min(Dis2Frs))])
            # if len(free_cluster_center) == 0:
            #     map_goal.append(self.agent_pos[e])
            #     print("cannot detect valid frontiers")
            # else:
            # # choose the frontier which is closest to the goal
            #     min_dis = 10000
            #     min_idx = -1
            #     for idx, fr in enumerate(free_cluster_center):
            #         dis = math.sqrt(math.hypot(fr[0]-goal[0], fr[1]-goal[1]))
            #         if dis < min_dis:
            #             min_dis = dis
            #             min_idx = idx
            #     map_goal.append(free_cluster_center[min_idx])

        # if self.visualization:
        #     self.visualize_map = copy.deepcopy(self.complete_map)
        #     for pt in self.visualize_goal:
        #         if pt[0] > 0 and pt[0] < 299 and pt[1] > 0 and pt[1] < 299:
        #             self.visualize_map[pt[0], pt[1]] = 128
        #             self.visualize_map[pt[0]-1, pt[1]] = 128
        #             self.visualize_map[pt[0]+1, pt[1]] = 128
        #             self.visualize_map[pt[0], pt[1]-1] = 128
        #             self.visualize_map[pt[0]-1, pt[1]-1] = 128
        #             self.visualize_map[pt[0]+1, pt[1]-1] = 128
        #             self.visualize_map[pt[0], pt[1]+1] = 128
        #             self.visualize_map[pt[0]-1, pt[1]+1] = 128
        #             self.visualize_map[pt[0]+1, pt[1]+1] = 128
        #         else:
        #             self.visualize_map[pt[0], pt[1]] = 128

        #     self.window.show_img(self.visualize_map)
        return np.array(map_goal)

    def step_for_cost(self):
        obs = []
        flag = False
        self.explored_each_map_t = []
        self.obstacle_each_map_t = []
        current_agent_pos = []
        each_agent_rewards = []
        self.num_step += 1
        reward_obstacle_each_map = np.zeros((self.num_agents, self.width, self.height))
        delta_reward_each_map = np.zeros((self.num_agents, self.width, self.height))
        reward_explored_each_map = np.zeros((self.num_agents, self.width, self.height))
        explored_all_map = np.zeros((self.width, self.height))
        obstacle_all_map = np.zeros((self.width, self.height))

        for i in range(self.num_agents):
            self.explored_each_map_t.append(np.zeros((self.width, self.height)))
            self.obstacle_each_map_t.append(np.zeros((self.width, self.height)))
        
        action = self.get_goal_for_cost()

        for i in range(self.num_agents): 
            robotGoal = action[i]
            if robotGoal[0] == self.agent_pos[i][0] and robotGoal[1] == self.agent_pos[i][1]:
                print("finish exploration")
                flag = True
                pass
            else:
                if self.gt_map[robotGoal[0], robotGoal[1]] == 254 and self.gt_map[self.agent_pos[i][0], self.agent_pos[i][1]] == 254:
                    global_plan = self.Astar_global_planner(self.agent_pos[i], robotGoal)   
                    pose = self.naive_local_planner(global_plan)
                    self.agent_pos[i] = pose[1][0]
                    self.path_log[i].append(self.agent_pos[i])
                    # import pdb; pdb.set_trace()
                    # self.agent_pos[i] = pose[-1][0]
                    # self.agent_dir[i] = pose[-1][1]
                    self.agent_dir[i] = random.randint(0, 3)
                    _, map_this_frame, (x_min, x_max), (y_min, y_max) = self.optimized_build_map(self.discrete_to_continuous(self.agent_pos[i]), 0, self.gt_map, self.resolution, self.sensor_range)  # can be modified to replace self.gt_map, self.resolution and self.sensor_range
                    self.built_map[i] = self.merge_two_map(self.built_map[i], map_this_frame, [x_min, x_max], [y_min, y_max])
                    # print("pose length: ", len(pose))
                    # start = time.time()
                    # self.build_map_given_path_for_multi_robot(pose, i)
                    # print("build map cost: ", time.time()-start)
                else:
                    print("Choose a non-free frontier")

        for i in range(self.num_agents): 
            # _, map_this_frame, _, _ = self.optimized_build_map(self.discrete_to_continuous(self.agent_pos[i]), 0, self.gt_map, self.resolution, self.sensor_range)
            # unknown: 205   free: 254   occupied: 0
            obs.append(self.built_map[i])
            current_agent_pos.append(self.agent_pos[i])
            self.explored_each_map_t[i] = (self.built_map[i] != 205).astype(int)
            self.obstacle_each_map_t[i] = (self.built_map[i] == 0).astype(int)

        for i in range(self.num_agents):
            self.explored_each_map[i] = np.maximum(self.explored_each_map[i], self.explored_each_map_t[i])
            self.obstacle_each_map[i] = np.maximum(self.obstacle_each_map[i], self.obstacle_each_map_t[i])
           
            reward_explored_each_map[i] = self.explored_each_map[i].copy()
            reward_explored_each_map[i][reward_explored_each_map[i] != 0] = 1
            
            reward_previous_explored_each_map = self.previous_explored_each_map[i].copy()
            reward_previous_explored_each_map[reward_previous_explored_each_map != 0] = 1

            # reward_obstacle_each_map[i] = self.obstacle_each_map[i].copy()
            # reward_obstacle_each_map[i][reward_obstacle_each_map[i] != 0] = 1

            delta_reward_each_map[i] = reward_explored_each_map[i]
            
            each_agent_rewards.append((np.array(delta_reward_each_map[i]) - np.array(reward_previous_explored_each_map)).sum())
            self.previous_explored_each_map[i] = self.explored_each_map[i]
        
        for i in range(self.num_agents):
            explored_all_map = np.maximum(self.explored_each_map[i], explored_all_map)
            obstacle_all_map = np.maximum(self.obstacle_each_map[i], obstacle_all_map)

        temp = explored_all_map + obstacle_all_map
        self.complete_map = np.zeros(temp.shape)
        self.complete_map[temp == 2] = 0
        self.complete_map[temp == 1] = 254
        self.complete_map[temp == 0] = 205

        explore_cell_size = np.sum((self.complete_map != 205).astype(int))
        if explore_cell_size / self.total_cell_size > 0.9:
            # compute time
            print("Path Length 90%: ", len(self.path_log[0]))
        
        if explore_cell_size / self.total_cell_size > 0.98:
            # compute time
            print("Path Length Total: ", len(self.path_log[0]))
            # std
            exploration_rate = []
            for e in range(self.num_agents):
                exploration_rate.append(np.sum((self.built_map[e] != 205).astype(int))/self.total_cell_size)
            print("std: ", np.std(np.array(exploration_rate)))
            print("overlap: ", np.sum(np.array(exploration_rate))-1)

        # reward_explored_all_map = explored_all_map.copy()
        # reward_explored_all_map[reward_explored_all_map != 0] = 1

        # delta_reward_all_map = reward_explored_all_map

        # reward_previous_all_map = self.previous_all_map.copy()
        # reward_previous_all_map[reward_previous_all_map != 0] = 1

        # merge_explored_reward = (np.array(delta_reward_all_map) - np.array(reward_previous_all_map)).sum()
        # self.previous_all_map = explored_all_map

        # info = {}
        # info['explored_all_map'] = np.array(explored_all_map)
        # info['current_agent_pos'] = np.array(current_agent_pos)
        # info['explored_each_map'] = np.array(self.explored_each_map)
        # info['obstacle_all_map'] = np.array(obstacle_all_map)
        # info['obstacle_each_map'] = np.array(self.obstacle_each_map)
        # info['agent_direction'] = np.array(self.agent_dir)
        # # info['agent_local_map'] = self.agent_local_map
        # if self.use_time_penalty:
        #     info['agent_explored_reward'] = np.array(each_agent_rewards) * 0.02 - 0.01
        #     info['merge_explored_reward'] = merge_explored_reward * 0.02 - 0.01
        # else:
        #     info['agent_explored_reward'] = np.array(each_agent_rewards) * 0.02
        #     info['merge_explored_reward'] = merge_explored_reward * 0.02
        # done = False
        # if delta_reward_all_map.sum() / self.total_cell_size >= self.target_ratio or flag:#(self.width * self.height)
        #     done = True       
        #     self.merge_ratio_step = self.num_step
        #     if self.use_complete_reward:
        #         info['merge_explored_reward'] += 0.1 * (delta_reward_all_map.sum() / self.total_cell_size)     
                
        # for i in range(self.num_agents):
        #     if delta_reward_each_map[i].sum() / self.total_cell_size >= self.target_ratio:#(self.width * self.height)
        #         self.agent_ratio_step[i] = self.num_step
        #         # if self.use_complete_reward:
        #         #     info['agent_explored_reward'][i] += 0.1 * (reward_explored_each_map[i].sum() / (self.width * self.height))
        
        # self.agent_reward = info['agent_explored_reward']
        # self.merge_reward = info['merge_explored_reward']
        # self.merge_ratio = delta_reward_all_map.sum() / self.total_cell_size #(self.width * self.height)
        # info['merge_explored_ratio'] = self.merge_ratio
        # info['merge_ratio_step'] = self.merge_ratio_step
        # for i in range(self.num_agents):
        #     info["agent{}_ratio_step".format(i)] = self.agent_ratio_step[i]

        # dones = np.array([done for agent_id in range(self.num_agents)])
        # if self.use_single_reward:
        #     rewards = 0.3 * np.expand_dims(info['agent_explored_reward'], axis=1) + 0.7 * np.expand_dims(np.array([info['merge_explored_reward'] for _ in range(self.num_agents)]), axis=1)
        # else:
        #     rewards = np.expand_dims(np.array([info['merge_explored_reward'] for _ in range(self.num_agents)]), axis=1)

        obs = np.array(obs)

        self.plot_map_with_path()
        # self.window.show_img(self.complete_map)
        # import pdb; pdb.set_trace()

        return obs
    
    def frontiers_detection_for_mmpf(self, map):
        '''
        detect frontiers from current built map
        '''
        obstacles = []
        frontiers = []
        height = map.shape[0]
        width = map.shape[1]
        for i in range(2, height-2):
            for j in range(2, width-2):
                if map[i][j] == 0:
                    obstacles.append([i,j])
                elif map[i][j] == 205:
                    numFree = 0
                    temp1 = 0
                    if map[i+1][j] == 254:
                        temp1 += 1 if map[i+2][j] == 254 else 0
                        temp1 += 1 if map[i+1][j+1] == 254 else 0
                        temp1 += 1 if map[i+1][j-1] == 254 else 0
                        numFree += (temp1 > 0)
                    if map[i][j+1] == 254:
                        temp1 += 1 if map[i][j+2] == 254 else 0
                        temp1 += 1 if map[i+1][j+1] == 254 else 0
                        temp1 += 1 if map[i-1][j+1] == 254 else 0
                        numFree += (temp1 > 0)     
                    if map[i-1][j] == 254:
                        temp1 += 1 if map[i-1][j+1] == 254 else 0
                        temp1 += 1 if map[i-1][j-1] == 254 else 0
                        temp1 += 1 if map[i-2][j] == 254 else 0
                        numFree += (temp1 > 0)
                    if map[i][j-1] == 254:
                        temp1 += 1 if map[i][j-2] == 254 else 0
                        temp1 += 1 if map[i+1][j-1] == 254 else 0
                        temp1 += 1 if map[i-1][j-1] == 254 else 0
                        numFree += (temp1 > 0)     
                    if numFree > 0:
                        frontiers.append([i,j])      
        return frontiers, obstacles

    def get_goal_for_mmpf(self):
        map_goal = []
        for e in range(self.num_agents):
            # goal = [int(self.width*data['global_goal'][e][0]), int(self.height*data['global_goal'][e][1])]
            # self.visualize_goal[e] = goal
            # occupancy_grid = data['global_obs'][e, 0] + data['global_obs'][e, 1]
            # obstacle: 2  unknown: 0   free: 1
            frs, _ = self.frontiers_detection_for_mmpf(self.complete_map)
            # cluster targets into different groups and find the center of each group.
            target_process = copy.deepcopy(frs)
            cluster_center = []
            infoGain_cluster = []

            path = []
            currentLoc = self.agent_pos[e]
            path.append(currentLoc)
            # path = []
            # currentLoc = self.continuous_to_discrete(self.robot.pos)
            # path.append(currentLoc)
            while(len(target_process) > 0):
                target_cluster = []
                target_cluster.append(target_process.pop())

                condition = True
                while(condition):
                    condition = False
                    size_target_process = len(target_process)
                    for i in reversed(range(size_target_process)):
                        for j in range(len(target_cluster)):
                            dis = abs(target_process[i][0] - target_cluster[j][0]) +  abs(target_process[i][1] - target_cluster[j][1])
                            if dis < 3:
                                target_cluster.append(target_process[i])
                                del target_process[i]
                                condition = True
                                break

                center_ = [0, 0]
                num_ = len(target_cluster)
                for i in range(num_):
                    center_[0] += target_cluster[i][0]
                    center_[1] += target_cluster[i][1]

                center_float = [float(center_[0])/float(num_), float(center_[1])/float(num_)]
                min_dis_ = 100.0
                min_idx_ = 10000
                for i in range(num_):
                    temp_dis_ = abs(center_float[0]-float(target_cluster[i][0])) + abs(center_float[1]-float(target_cluster[i][1]))
                    if temp_dis_ < min_dis_:
                        min_dis_ = temp_dis_
                        min_idx_ = i

                cluster_center.append([target_cluster[min_idx_][0], target_cluster[min_idx_][1]])
                infoGain_cluster.append(num_)
            # free_cluster_center = []
            # for i in range(len(cluster_center)):
            #     # find the nearest free grid
            #     for x in range(3):
            #         for y in range(3):
            #             if self.built_map[e][cluster_center[i][0]-1+x, cluster_center[i][1]-1+y] == 254:
            #                 free_cluster_center.append([cluster_center[i][0]-1+x, cluster_center[i][1]-1+y])
            #                 break
            #         else:
            #             continue
            #         break
            K_ATTRACT = 1
            riverFlowPotentialGain = 1

            cluster_num = len(cluster_center)

            dismap_target = []

            for i in range(cluster_num):
                dismap_target.append(self.dismapConstruction_start_target(cluster_center[i], self.built_map[e]))

            # calculate path
            iteration = 1
            currentPotential = 10000
            riverFlowPotentialGain = 1
            minDis2Frontier  = 10000;  
            while (iteration < 3000 and minDis2Frontier > 1):
                potential = [0,0,0,0]
                min_idx = -1
                min_potential = 10000
                loc_around = [[currentLoc[0]-1, currentLoc[1]], # upper
                            [currentLoc[0], currentLoc[1]-1], # left
                            [currentLoc[0]+1, currentLoc[1]], # down
                            [currentLoc[0], currentLoc[1]+1]] # right
                for i in range(4):
                    curr_around = loc_around[i]
                    # calculate current potential
                    attract = 0
                    repulsive = 0
                    for j in range(len(cluster_center)):
                        temp = dismap_target[j][curr_around[0],curr_around[1]]
                        if temp < 1:
                            continue
                        attract = attract - K_ATTRACT*infoGain_cluster[j]/temp

                    # to increase the potential if currend point has been passed before
                    for j in range(len(path)):
                        if curr_around[0] == path[j][0] and curr_around[1] == path[j][1]:
                            attract += riverFlowPotentialGain*5

                    # Add impact of robots.
                    # import random
                    # if random.random() > 0.5:
                    #     for r_num in range(self.num_agents):
                    #         if r_num != e: 
                    #             dis_ = abs(self.agent_pos[r_num][0]-curr_around[0])+abs(self.agent_pos[r_num][1]-curr_around[1])
                    #             temp_ = -dis_
                    #             if dis_ < 50:
                    #                 attract += 0.01*temp_

                    potential[i] = attract
                    if min_potential > potential[i]:
                        min_potential = potential[i]
                        min_idx = i

                if currentPotential > min_potential:
                    path.append(loc_around[min_idx])
                    currentPotential = min_potential
                else:
                    riverFlowPotentialGain += 1

                currentLoc = path[-1]

                for i in range(len(cluster_center)):
                    temp_dis_ = dismap_target[i][currentLoc[0],currentLoc[1]]
                    if temp_dis_ == 0 and abs(currentLoc[0]-cluster_center[i][0]) + abs(currentLoc[1]-cluster_center[i][1]) > 0:
                        continue
                    if minDis2Frontier > temp_dis_:
                        minDis2Frontier = temp_dis_
                iteration += 1

            # robotGoal = self.discrete_to_continuous(path[-1])
            
            map_goal.append(path[-1])
        # import pdb; pdb.set_trace()
        return np.array(map_goal)

    def step_for_mmpf(self):
        obs = []
        flag = False
        self.explored_each_map_t = []
        self.obstacle_each_map_t = []
        current_agent_pos = []
        each_agent_rewards = []
        self.num_step += 1
        reward_obstacle_each_map = np.zeros((self.num_agents, self.width, self.height))
        delta_reward_each_map = np.zeros((self.num_agents, self.width, self.height))
        reward_explored_each_map = np.zeros((self.num_agents, self.width, self.height))
        explored_all_map = np.zeros((self.width, self.height))
        obstacle_all_map = np.zeros((self.width, self.height))

        for i in range(self.num_agents):
            self.explored_each_map_t.append(np.zeros((self.width, self.height)))
            self.obstacle_each_map_t.append(np.zeros((self.width, self.height)))
        
        action = self.get_goal_for_mmpf()

        for i in range(self.num_agents): 
            robotGoal = action[i]
            if robotGoal[0] == self.agent_pos[i][0] and robotGoal[1] == self.agent_pos[i][1]:
                print("finish exploration")
                flag = True
                pass
            else:
                if self.gt_map[robotGoal[0], robotGoal[1]] == 254 and self.gt_map[self.agent_pos[i][0], self.agent_pos[i][1]] == 254:
                    global_plan = self.Astar_global_planner(self.agent_pos[i], robotGoal)   
                    pose = self.naive_local_planner(global_plan)
                    self.agent_pos[i] = pose[1][0]
                    self.path_log[i].append(self.agent_pos[i])
                    # import pdb; pdb.set_trace()
                    # self.agent_pos[i] = pose[-1][0]
                    # self.agent_dir[i] = pose[-1][1]
                    self.agent_dir[i] = random.randint(0, 3)
                    _, map_this_frame, (x_min, x_max), (y_min, y_max) = self.optimized_build_map(self.discrete_to_continuous(self.agent_pos[i]), 0, self.gt_map, self.resolution, self.sensor_range)  # can be modified to replace self.gt_map, self.resolution and self.sensor_range
                    self.built_map[i] = self.merge_two_map(self.built_map[i], map_this_frame, [x_min, x_max], [y_min, y_max])
                    # print("pose length: ", len(pose))
                    # start = time.time()
                    # self.build_map_given_path_for_multi_robot(pose, i)
                    # print("build map cost: ", time.time()-start)
                else:
                    print("Choose a non-free frontier")

        for i in range(self.num_agents): 
            # _, map_this_frame, _, _ = self.optimized_build_map(self.discrete_to_continuous(self.agent_pos[i]), 0, self.gt_map, self.resolution, self.sensor_range)
            # unknown: 205   free: 254   occupied: 0
            obs.append(self.built_map[i])
            current_agent_pos.append(self.agent_pos[i])
            self.explored_each_map_t[i] = (self.built_map[i] != 205).astype(int)
            self.obstacle_each_map_t[i] = (self.built_map[i] == 0).astype(int)

        for i in range(self.num_agents):
            self.explored_each_map[i] = np.maximum(self.explored_each_map[i], self.explored_each_map_t[i])
            self.obstacle_each_map[i] = np.maximum(self.obstacle_each_map[i], self.obstacle_each_map_t[i])
           
            reward_explored_each_map[i] = self.explored_each_map[i].copy()
            reward_explored_each_map[i][reward_explored_each_map[i] != 0] = 1
            
            reward_previous_explored_each_map = self.previous_explored_each_map[i].copy()
            reward_previous_explored_each_map[reward_previous_explored_each_map != 0] = 1

            # reward_obstacle_each_map[i] = self.obstacle_each_map[i].copy()
            # reward_obstacle_each_map[i][reward_obstacle_each_map[i] != 0] = 1

            delta_reward_each_map[i] = reward_explored_each_map[i]
            
            each_agent_rewards.append((np.array(delta_reward_each_map[i]) - np.array(reward_previous_explored_each_map)).sum())
            self.previous_explored_each_map[i] = self.explored_each_map[i]
        
        for i in range(self.num_agents):
            explored_all_map = np.maximum(self.explored_each_map[i], explored_all_map)
            obstacle_all_map = np.maximum(self.obstacle_each_map[i], obstacle_all_map)

        temp = explored_all_map + obstacle_all_map
        self.complete_map = np.zeros(temp.shape)
        self.complete_map[temp == 2] = 0
        self.complete_map[temp == 1] = 254
        self.complete_map[temp == 0] = 205

        explore_cell_size = np.sum((self.complete_map != 205).astype(int))
        if explore_cell_size / self.total_cell_size > 0.9:
            # compute time
            print("Path Length 90%: ", len(self.path_log[0]))
        
        if explore_cell_size / self.total_cell_size > 0.98:
            # compute time
            print("Path Length Total: ", len(self.path_log[0]))
            # std
            exploration_rate = []
            for e in range(self.num_agents):
                exploration_rate.append(np.sum((self.built_map[e] != 205).astype(int))/self.total_cell_size)
            print("std: ", np.std(np.array(exploration_rate)))
            print("overlap: ", np.sum(np.array(exploration_rate))-1)

        # reward_explored_all_map = explored_all_map.copy()
        # reward_explored_all_map[reward_explored_all_map != 0] = 1

        # delta_reward_all_map = reward_explored_all_map

        # reward_previous_all_map = self.previous_all_map.copy()
        # reward_previous_all_map[reward_previous_all_map != 0] = 1

        # merge_explored_reward = (np.array(delta_reward_all_map) - np.array(reward_previous_all_map)).sum()
        # self.previous_all_map = explored_all_map

        # info = {}
        # info['explored_all_map'] = np.array(explored_all_map)
        # info['current_agent_pos'] = np.array(current_agent_pos)
        # info['explored_each_map'] = np.array(self.explored_each_map)
        # info['obstacle_all_map'] = np.array(obstacle_all_map)
        # info['obstacle_each_map'] = np.array(self.obstacle_each_map)
        # info['agent_direction'] = np.array(self.agent_dir)
        # # info['agent_local_map'] = self.agent_local_map
        # if self.use_time_penalty:
        #     info['agent_explored_reward'] = np.array(each_agent_rewards) * 0.02 - 0.01
        #     info['merge_explored_reward'] = merge_explored_reward * 0.02 - 0.01
        # else:
        #     info['agent_explored_reward'] = np.array(each_agent_rewards) * 0.02
        #     info['merge_explored_reward'] = merge_explored_reward * 0.02
        # done = False
        # if delta_reward_all_map.sum() / self.total_cell_size >= self.target_ratio or flag:#(self.width * self.height)
        #     done = True       
        #     self.merge_ratio_step = self.num_step
        #     if self.use_complete_reward:
        #         info['merge_explored_reward'] += 0.1 * (delta_reward_all_map.sum() / self.total_cell_size)     
                
        # for i in range(self.num_agents):
        #     if delta_reward_each_map[i].sum() / self.total_cell_size >= self.target_ratio:#(self.width * self.height)
        #         self.agent_ratio_step[i] = self.num_step
        #         # if self.use_complete_reward:
        #         #     info['agent_explored_reward'][i] += 0.1 * (reward_explored_each_map[i].sum() / (self.width * self.height))
        
        # self.agent_reward = info['agent_explored_reward']
        # self.merge_reward = info['merge_explored_reward']
        # self.merge_ratio = delta_reward_all_map.sum() / self.total_cell_size #(self.width * self.height)
        # info['merge_explored_ratio'] = self.merge_ratio
        # info['merge_ratio_step'] = self.merge_ratio_step
        # for i in range(self.num_agents):
        #     info["agent{}_ratio_step".format(i)] = self.agent_ratio_step[i]

        # dones = np.array([done for agent_id in range(self.num_agents)])
        # if self.use_single_reward:
        #     rewards = 0.3 * np.expand_dims(info['agent_explored_reward'], axis=1) + 0.7 * np.expand_dims(np.array([info['merge_explored_reward'] for _ in range(self.num_agents)]), axis=1)
        # else:
        #     rewards = np.expand_dims(np.array([info['merge_explored_reward'] for _ in range(self.num_agents)]), axis=1)

        obs = np.array(obs)

        self.plot_map_with_path()
        # self.window.show_img(self.complete_map)
        # import pdb; pdb.set_trace()

        return obs
    
    # For cost-based exploration (always go to the nearest frontier)
    def cost_based_explore(self, window_path, window_frontiers):
        path_log = []
        while(True):
            targets, obstacles = self.frontiers_detection()

            # cluster targets into different groups and find the center of each group.
            target_process = copy.deepcopy(targets)
            cluster_center = []
            infoGain_cluster = []

            path = []
            currentLoc = self.continuous_to_discrete(self.robot.pos)
            path.append(currentLoc)

            while(len(target_process) > 0):
                target_cluster = []
                target_cluster.append(target_process.pop())

                condition = True
                while(condition):
                    condition = False
                    size_target_process = len(target_process)
                    for i in reversed(range(size_target_process)):
                        for j in range(len(target_cluster)):
                            dis = abs(target_process[i][0] - target_cluster[j][0]) +  abs(target_process[i][1] - target_cluster[j][1])
                            if dis < 3:
                                target_cluster.append(target_process[i])
                                del target_process[i]
                                condition = True
                                break

                center_ = [0, 0]
                num_ = len(target_cluster)
                for i in range(num_):
                    center_[0] += target_cluster[i][0]
                    center_[1] += target_cluster[i][1]

                center_float = [float(center_[0])/float(num_), float(center_[1])/float(num_)]
                min_dis_ = 100.0
                min_idx_ = 10000
                for i in range(num_):
                    temp_dis_ = abs(center_float[0]-float(target_cluster[i][0])) + abs(center_float[1]-float(target_cluster[i][1]))
                    if temp_dis_ < min_dis_:
                        min_dis_ = temp_dis_
                        min_idx_ = i

                cluster_center.append([target_cluster[min_idx_][0], target_cluster[min_idx_][1]])
                infoGain_cluster.append(num_)

            vis_map = self.visualize_mmpf_frontiers(cluster_center)
            window_frontiers.show_img(vis_map) 

            curr_dismap = self.dismapConstruction_start_target(currentLoc)


            # window = Window('dismap')
            # window.show(block=False)
            # window.show_img(curr_dismap)
            # import pdb; pdb.set_trace()

            Dis2Frs = []
            free_cluster_center = []
            for i in range(len(cluster_center)):
                # find the nearest free grid
                for x in range(3):
                    for y in range(3):
                        if self.built_map[cluster_center[i][0]-1+x, cluster_center[i][1]-1+y] == 254:
                            Dis2Frs.append(curr_dismap[cluster_center[i][0]-1+x, cluster_center[i][1]-1+y])
                            free_cluster_center.append([cluster_center[i][0]-1+x, cluster_center[i][1]-1+y])
                            break
                    else:
                        continue
                    break
            
            robotGoal = free_cluster_center[Dis2Frs.index(min(Dis2Frs))]
            robotGoal = self.discrete_to_continuous(robotGoal)
            global_plan = self.Astar_global_planner(self.robot.pos, robotGoal)   
            pose = self.naive_local_planner(global_plan)
            path_log.extend(global_plan)
            # self.visualize_rrt(V)
            # self.visualize_rrt_frontiers(self.frontiers)
            self.build_map_given_path(pose)
            self.robot.pos = self.discrete_to_continuous(pose[-1][0])

            vis_map = self.plot_path(path_log, self.built_map)
            window_path.show_img(vis_map)
            import pdb; pdb.set_trace()

    # For utility-based exploration (always go to the largest InfoGain frontier)
    def utility_based_explore(self, window_path, window_frontiers):
        path_log = []
        while(True):
            targets, obstacles = self.frontiers_detection()

            # cluster targets into different groups and find the center of each group.
            target_process = copy.deepcopy(targets)
            cluster_center = []
            infoGain_cluster = []

            path = []
            currentLoc = self.continuous_to_discrete(self.robot.pos)
            path.append(currentLoc)

            while(len(target_process) > 0):
                target_cluster = []
                target_cluster.append(target_process.pop())

                condition = True
                while(condition):
                    condition = False
                    size_target_process = len(target_process)
                    for i in reversed(range(size_target_process)):
                        for j in range(len(target_cluster)):
                            dis = abs(target_process[i][0] - target_cluster[j][0]) +  abs(target_process[i][1] - target_cluster[j][1])
                            if dis < 3:
                                target_cluster.append(target_process[i])
                                del target_process[i]
                                condition = True
                                break

                center_ = [0, 0]
                num_ = len(target_cluster)
                for i in range(num_):
                    center_[0] += target_cluster[i][0]
                    center_[1] += target_cluster[i][1]

                center_float = [float(center_[0])/float(num_), float(center_[1])/float(num_)]
                min_dis_ = 100.0
                min_idx_ = 10000
                for i in range(num_):
                    temp_dis_ = abs(center_float[0]-float(target_cluster[i][0])) + abs(center_float[1]-float(target_cluster[i][1]))
                    if temp_dis_ < min_dis_:
                        min_dis_ = temp_dis_
                        min_idx_ = i

                cluster_center.append([target_cluster[min_idx_][0], target_cluster[min_idx_][1]])
                infoGain_cluster.append(num_)

            vis_map = self.visualize_mmpf_frontiers(cluster_center)
            window_frontiers.show_img(vis_map) 

            max_infoGain = 0
            max_idx = -1
            for i in range(len(infoGain_cluster)):
                if infoGain_cluster[i] > max_infoGain:
                    max_infoGain = infoGain_cluster[i]
                    max_idx = i

            for x in range(3):
                for y in range(3):
                    if self.built_map[cluster_center[max_idx][0]-1+x, cluster_center[max_idx][1]-1+y] == 254:
                        robotGoal = [cluster_center[max_idx][0]-1+x, cluster_center[max_idx][1]-1+y]
                        break
                else:
                    continue
                break
            
            robotGoal = self.discrete_to_continuous(robotGoal)
            global_plan = self.Astar_global_planner(self.robot.pos, robotGoal)   
            pose = self.naive_local_planner(global_plan)
            path_log.extend(global_plan)
            # self.visualize_rrt(V)
            # self.visualize_rrt_frontiers(self.frontiers)
            self.build_map_given_path(pose)
            self.robot.pos = self.discrete_to_continuous(pose[-1][0])

            vis_map = self.plot_path(path_log, self.built_map)
            window_path.show_img(vis_map)
            import pdb; pdb.set_trace()     

class Robot(object):
    def __init__(self, pos, orn, linear_velocity=0, angular_velocity=0):
        self.pos = pos
        self.orn = orn
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity
        
        # Kinematic constraints
        self.max_a = 0.2  # Max longitudinal acceleration
        self.max_d_w = 0.7  # Max change in Yaw rate
        self.max_v = 1.0  # Max linear velocity
        self.min_v = 0  # Min linear velocity
        self.max_w = 0.8  # Max angular velocity
        self.min_w = -0.8 # Min angular velocity
        
        # Discretization
        self.dt = 0.1 
        self.v_res = 0.05  # Linear velocity resolution
        self.w_res = 0.02  # Angular velocity resolution

    def update(self, t):
        r = self.linear_velocity / self.angular_velocity
        delta_theta = self.angular_velocity * t
        delta_x = r*math.cos(self.orn) - r*math.cos(self.orn+delta_theta)
        delta_y = r*math.sin(self.orn+delta_theta)-r*math.sin(self.orn)
        self.pos[0] = self.pos[0] - delta_x
        self.pos[1] = self.pos[1] + delta_y
        self.orn = self.orn + delta_theta

class DWA:
    # String literals 
    def __init__(self, max_v, min_v, max_w, min_w, dt, dw_time, gt_map):
        self.config_params = robot_config.Parameters()
        self.max_v = max_v  # Max linear velocity
        self.min_v = min_v  # Min linear velocity
        self.max_w = max_w  # Max angular velocity
        self.min_w = min_w # Min angular velocity
        self.dt = dt
        self.dw_time = dw_time
        self.heading_gain = 0.2
        self.velocity_gain = 1.0
        self.distance_gain = 1.7
        self.max_a = 0.2
        self.max_d_w = 0.7
        self.gt_map = gt_map

    # I should add the type of the incoming arguments 
    def update_motion(self, x, u):
        """
        Update the robot motion based on the control commads (linear
        and angular velocity)

        Keyword arguments:
        x -- numpy array containing the current robot state -> [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        u -- numpy array containing the control commands 
        """
        # Yaw rate update
        x[2] += u[1] * self.dt

        # X and Y positions update
        x[0] += u[0] * math.cos(x[2]) * self.dt
        x[1] += u[0] * math.sin(x[2]) * self.dt
        
        # Linear and Angular velocities update
        # x[3] = u[0]
        # x[4] = u[1]

        # Return the updated state
        return x

    # I should add the type of the incoming arguments 
    def calculate_dw(self, u):
        """
        Calculate the dynamic window, equivalent to the velocities that can be
        reached within the next time interval (dt). Accerelations are considered
        in this calculation as they are the hardware limitation.  
        
        Note that the dynamic window is centred around the actual velocity and 
        the extensions of it depend on the accelerations that can be exerted. All 
        curvatures outside the dynamic window cannot be reached within the next 
        time interval and thus are not considered for the obstacle avoidance.

        Keyword arguments:
        x -- numpy array containing the current robot state -> [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        u -- numpy array containing the control commands -> [v(m/s), omega(rad/s)]
        """
        # We need to make sure we are within the motion constraints, in other words we need to 
        # lie within the dynamic window, for that reason we consider the velocity boundaries
        dw = [
            max(u[0] - self.max_a * self.dt, self.min_v), # Min lin. vel 
            min(u[0] + self.max_a * self.dt, self.max_v), # Max lin. vel
            max(u[1] - self.max_d_w * self.dt, self.min_w), # Min ang. vel
            min(u[1] + self.max_d_w * self.dt, self.max_w) # Max ang. vel
        ]

        # Dynamic window -> [V_min, V_max, W_min, W_max]  
        return dw

    def calculate_traj(self, x, v, w):
        """
        Calculate a possible trajectory in a given time window. The calculation
        considers the currrent robot state and a single control input (As it assumes
        they are constants).  

        Keyword arguments:
        x -- numpy array containing the current robot state -> [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        v -- linear velocity for trajectory prediction
        w -- angular velocity for trajectory prediction  
        """
        # Temporal vector state
        x_tmp = np.array(x)
        # Array for saving the trajectory
        predicted_traj = np.array(x_tmp)
        # Control input
        u = np.array([v, w])
        time = 0.0

        # Prediction window (Lower than 3.0 seconds)
        while(time <= self.dw_time):
            # Update the motion and save the value for the trajectory
            x_tmp = self.update_motion(x_tmp, u)
            # Append the predicted state to the trajectory
            predicted_traj = np.vstack((predicted_traj, x_tmp))
            time += self.dt

        # Predicted trajectory array containing the trajectory as -> [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        # for each time step in the next three seconds.
        return predicted_traj
        
    def target_heading(self, trajectory, goal):
        """
        Calcutes the heading cost between the current heading angle and the goal, 
        it is maximum when the robot is moving directly towards the target.

        Keyword arguments:
        trajectory -- numpy array containing the predicted trajectory
        """
        pos_x = trajectory[-1, 0]  # Extract the last X position in the trajectory 
        pos_y = trajectory[-1, 1]  # Extract the last Y position in the trajectory

        diff_x = goal[0] - pos_x
        diff_y = goal[1] - pos_y

        heading = math.atan2(diff_y, diff_x)

        # Calculate the difference or error between the trajectory angle and the the current yaw
        error = heading - trajectory[-1, 2]
        return abs(math.atan2(math.sin(error), math.cos(error)))

    def obstacle_distance(self, trajectory):
        """
        Calculate the cost (Distance) to the closest obstacle in the road. When this 
        distance (Between the robot and obstacle) is small the robot will tend to move
        around it.

        Keyword arguments:
        trajectory -- numpy array containing the predicted trajectory
        """
        obst_x = self.config_params.obstacles[:, 0]
        obst_y = self.config_params.obstacles[:, 1]

        """
        Note:
        Keep in mind that the following operation returns an array for 
        each substraction. For example if there are 5 obstacles we will 
        calculate 5 different matrices between the X trajectory and the 
        5 obstacles positions in X, similarly this will take place in Y.

        traj[:, 0] = [1, 2, 3, 4, 5] -> Trajectory in X
        obs_x[:, None] = [1.5, 2.5, 3.1, 3.6, 4.0] -> Obstacles position in X
        traj[:, 1] - obs_x[:, None] = [
            [-0.5, 0.5, 1.5, 2.5, 3.5], -> X and first obstacle
            [-1.5, -0.5, 0.5, 1.5, 2.5], -> X and second obstacle
            [-2.1, -1.1, -0.1, 0.9, 1.9], -> X and third obstacle
            ...
            ]
        """
        # Calculate the difference between the trajectory and obstacles
        diff_x = trajectory[:, 0] - obst_x[:, None] # Broadcasting
        diff_y = trajectory[:, 1] - obst_y[:, None] # Broadcasting
        
        """
        Numpy hypot takes the distance of each point of the trajectory 
        to one given osbtacle and calculates the euclidean distance for 
        each individual point in the trajectory. Similarly we will have 
        the same number of vectors for each obstacle in the space. 
        
        Result will be an array contanining the euclidean distance from 
        each trajectory point to each of the obstacles, hence we will 
        have n array where n is given by the number of obstacles.
        """
        # Calculate the euclidean distance between trajectory and obstacles
        dist = np.array(np.hypot(diff_x, diff_y))

        # Some point in the whole array finds an obstacle
        if np.any(dist <= self.config_params.chassis_radius * 0.5):
            # Hit ane obstacle, hence discard this trajectory
            return float("inf")

        # We can either return the cost or the distance
        min_value = np.min(dist)

        # The closer we are the greater this number as the goal is to maximize
        # the cost function.
        return 1 / min_value

    def calculate_ctrl_traj(self, x, goal):
        """
        Calculates the optimal control input and trajectory from a given robot 
        state. This function performs the calculation of the dynamic window, the
        trajectory generation and the cost function calculation.

        It traverse all the possible trajectories and then return the optimal one.

        Keyword arguments:
        x -- numpy array containing the current robot state -> [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        """
        # Calculate the admissible velocities (DW) - [V_min, V_max, W_min, W_max] 
        dw_v = self.calculate_dw(x)

        # Array of velocities (Initial, Final, Resolution)
        v_vect = np.arange(dw_v[0], dw_v[1], self.config_params.v_res)
        w_vect = np.arange(dw_v[2], dw_v[3], self.config_params.w_res)

        minimum_cost = float("inf")

        optimal_u = [0.0, 0.0]
        optimal_traj = x

        # Iterate through the linear velocties
        for v in v_vect:
            # Iterate through the angular velocties
            for w in w_vect:
                # Trajectory generation for the following three seconds
                trajectory = self.calculate_traj(x, v, w)

                # Obstacle distance cost
                clearance_cost = self.config_params.distance_gain * self.obstacle_distance(trajectory) 
                # Heading angle cost
                heading_cost = self.config_params.heading_gain * self.target_heading(trajectory, goal)
    
                # Velocity cost. Difference between max velocity and final trajectory velocity
                velocity_cost = self.config_params.velocity_gain * abs(self.config_params.max_v - trajectory[-1, 3])

                total_cost = clearance_cost + heading_cost + velocity_cost

                if (minimum_cost >= total_cost):
                    # Set the new optimal cost
                    minimum_cost = total_cost
                    # Set the optimal control input
                    optimal_u = [v, w]
                    # Set the optimal trajectory
                    optimal_traj = trajectory
                    
                    # Both veloties are too small, we run into troublew if it happens
                    """
                    This is a particular scenario. It is not part of the algorithm, but
                    should be considered to find a solution. There are some scenarios
                    where the robot linear control velocity is too small as well as the 
                    current angular velocity and additionally it is moving toward an 
                    obstacle (Get stuck close to the obstacle), so we add some angular
                    velocity to force it to turn around the obstacle.
                    """
                    if (abs(optimal_u[0]) < 0.001 and abs(x[3]) < 0.001):
                        # We added some angular velocity so the robot turn around the obstacles  
                        # and calculates a new linear velocity
                        optimal_u[1] = -40.0 * math.pi / 180.0 
                        # At this point we can consider the heading angle and add or substract this
                        # angular velocity

        return optimal_u, optimal_traj


def check_occupied(i, j, map):
    '''
    check whether [i, j] pixel is occupied
    205--unknown
    254--free space
    0--occupied
    '''
    feasibility = True
    if map[i,j] == 0:
        feasibility = False
    return feasibility

def check_visible(i, j, map):
    '''
    check whether [i, j] pixel is visible from the origin of map
    '''
    step_length = max(abs(i - map.shape[0]/2),
                      abs(j - map.shape[1]/2))
    path_x = np.linspace(i, map.shape[0]/2, int(step_length)+2)
    path_y = np.linspace(j, map.shape[1]/2, int(step_length)+2)
    for i in range(1, int(step_length)+1):
        # print(int(math.floor(path_x[i])))
        # print(int(math.floor(path_y[i])))
        if (not check_occupied(int(math.floor(path_x[i])), int(math.floor(path_y[i])), map)):
            return False
    return True

def check_visible_test(i, j, map):
    '''
    check whether [i, j] pixel is visible from the origin of map
    '''
    laser_path = []
    step_length = max(abs(i - map.shape[0]/2),
                      abs(j - map.shape[1]/2))
    path_x = np.linspace(i, map.shape[0]/2, int(step_length)+2)
    path_y = np.linspace(j, map.shape[1]/2, int(step_length)+2)
    for i in range(1, int(step_length)+1):
        # print(int(math.floor(path_x[i])))
        # print(int(math.floor(path_y[i])))
        laser_path.append([int(math.floor(path_x[i])), int(math.floor(path_y[i]))])
    return laser_path

def build_map_per_frame(pos, orn, gt_map, resolution, sensor_range):
    '''
    build map from a specific pos with a omni-directional scan
    pos: [x, y]
    '''
    grid_range = int(sensor_range / resolution)
    # width = 6*grid_range
    # height = 6*grid_range
    height = gt_map.shape[0]
    width = gt_map.shape[1]
    init_map = np.zeros((width, height))
    x, y = int(height/2+pos[0]-grid_range), int(width/2+pos[1]-grid_range) # relative to the upper left corner of the picture
    init_map = gt_map[x:x+2*grid_range, y:y+2*grid_range]
    # debug: show the robot's pos
    # gt_map[int(height/2+pos[0]), int(width/2+pos[1])] = 0
    # img = Image.fromarray(gt_map.astype('uint8'))
    # img.save("debug.pgm")
    # img.show()
    # import pdb; pdb.set_trace()
    # put mask
    mask_map = copy.deepcopy(init_map)
    mask_map_origin = [mask_map.shape[0]/2, mask_map.shape[1]/2]
    # debug: show the robot's pos
    # mask_map[int(mask_map.shape[0]/2), int(mask_map.shape[1]/2)] = 0
    # img = Image.fromarray(mask_map.astype('uint8'))
    # img.save("debug_mask.pgm")
    # img.show()
    # import pdb; pdb.set_trace()
    for i in range(mask_map.shape[0]):
        for j in range(mask_map.shape[1]):
            dist = math.sqrt((i+0.5-mask_map_origin[0])*(i+0.5-mask_map_origin[0])+(j+0.5-mask_map_origin[1])*(j+0.5-mask_map_origin[1]))
            # unknown
            if dist > grid_range:
                mask_map[i][j] = 205
            else:
                # print(j)
                # left upper
                flag = True
                if i + 0.5 - mask_map.shape[0]/2 < -1 and j + 0.5 - mask_map.shape[1]/2 < -1:
                    flag = check_visible(i+1, j+1, init_map)
                # left down
                elif i + 0.5 - mask_map.shape[0]/2 > 1 and j + 0.5 - mask_map.shape[1]/2 < -1:
                    flag = check_visible(i, j+1, init_map)
                # right upper
                elif i + 0.5 - mask_map.shape[0]/2 < -1 and j + 0.5 - mask_map.shape[1]/2 > 1:
                    flag = check_visible(i+1, j, init_map)
                # right down
                elif i + 0.5 - mask_map.shape[0]/2 > 1 and j + 0.5 - mask_map.shape[1]/2 > 1:
                    flag = check_visible(i, j, init_map)
                elif i - mask_map.shape[0]/2 == -1 and j - mask_map.shape[1]/2 < -1:
                    flag = check_visible(i, j+1, init_map)
                elif i - mask_map.shape[0]/2 == -1 and j - mask_map.shape[1]/2 > 1:
                    flag = check_visible(i, j, init_map)
                elif i - mask_map.shape[0]/2 == 0 and j - mask_map.shape[1]/2 < -1:
                    flag = check_visible(i+1, j+1, init_map)
                elif i - mask_map.shape[0]/2 == 0 and j - mask_map.shape[1]/2 > 1:
                    flag = check_visible(i+1, j, init_map)
                elif i - mask_map.shape[0]/2 < -1 and j - mask_map.shape[1]/2 == -1:
                    flag = check_visible(i+1, j, init_map)
                elif i - mask_map.shape[0]/2 > 1 and j - mask_map.shape[1]/2 == -1:
                    flag = check_visible(i, j, init_map)
                elif i - mask_map.shape[0]/2 < -1 and j - mask_map.shape[1]/2 == 0:
                    flag = check_visible(i+1, j+1, init_map)
                elif i - mask_map.shape[0]/2 > 1 and j - mask_map.shape[1]/2 == 0:
                    flag = check_visible(i, j+1, init_map)
                if not flag:
                    mask_map[i][j] = 205
    scale_map = np.zeros(gt_map.shape)
    scale_map[:,:] = 205
    scale_map[x:x+2*grid_range, y:y+2*grid_range] = mask_map
    return mask_map, scale_map

def merge_two_map(map1, map2):
    '''
    merge two map into one map
    '''
    merge_map = map1 + map2
    for i in range(merge_map.shape[0]):
        for j in range(merge_map.shape[1]):
            if merge_map[i][j] == 0 or merge_map[i][j] == 205 or merge_map[i][j] == 254:
                merge_map[i][j] = 0
            elif merge_map[i][j] == 410:
                merge_map[i][j] = 205
            elif merge_map[i][j] == 459 or merge_map[i][j] == 508:
                merge_map[i][j] = 254
    return merge_map

def build_map_along_traj(traj, gt_map, resolution, sensor_range):
    '''
    build map along a given traj (only depend on robot's pos)
    '''
    traj_length = len(traj)
    _, traj_map = build_map_per_frame(traj[0], 0, gt_map, resolution, sensor_range)
    for pos in traj[1:]:
        _, temp_map = build_map_per_frame(pos, 0, gt_map, resolution, sensor_range)
        traj_map = merge_two_map(traj_map, temp_map)
    return traj_map

def visualize_path(traj, map, save_file=None):
    '''
    plot the traj in the map
    '''
    vis_map = copy.deepcopy(map)
    for pos in traj:
        vis_map[int(vis_map.shape[0]/2+pos[0]), int(vis_map.shape[1]/2+pos[1])] = 127
    img = Image.fromarray(vis_map.astype('uint8'))
    # img.show()
    if save_file:
        img.save(save_file)

def redraw(img):
    window.show_img(img)

def reset():
    obs = env.reset()

    redraw(obs)

def step():
    obs = env.step_by_teleop()
    vis = env.visualize_path()
    redraw(vis)

def key_handler(event):
    print("Robot state: ", env.robot.pos, ' ', env.robot.orn)
    print("linear velocity: ", env.robot.linear_velocity, "    angular velocity: ", env.robot.angular_velocity)
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        print("Local Planner...Robot moves toward the global goal.")
        optimal_u, optimal_traj = env.dwa_planner(env.robot.pos, env.robot.orn, global_path)
        env.step(optimal_u)
        vis = env.visualize_path()
        redraw(vis)
        # step()
        return
    if event.key == 'right':
        print("Robot state: ", env.robot.pos, ' ', env.robot.orn)
        env.robot.angular_velocity -= ANG_VEL_STEP_SIZE
        env.robot.angular_velocity = constrain(env.robot.angular_velocity, -BURGER_MAX_ANG_VEL, BURGER_MAX_ANG_VEL)
        print("linear velocity: ", env.robot.linear_velocity, "    angular velocity: ", env.robot.angular_velocity)
        step()
        return
    if event.key == 'up':
        print("Robot state: ", env.robot.pos, ' ', env.robot.orn)
        env.robot.linear_velocity += LIN_VEL_STEP_SIZE
        env.robot.linear_velocity = constrain(env.robot.linear_velocity, -BURGER_MAX_LIN_VEL, BURGER_MAX_LIN_VEL)
        print("linear velocity: ", env.robot.linear_velocity, "    angular velocity: ", env.robot.angular_velocity)
        step()
        return
    if event.key == 'down':
        print("Robot state: ", env.robot.pos, ' ', env.robot.orn)
        env.robot.linear_velocity -= LIN_VEL_STEP_SIZE
        env.robot.linear_velocity = constrain(env.robot.linear_velocity, -BURGER_MAX_LIN_VEL, BURGER_MAX_LIN_VEL)
        print("linear velocity: ", env.robot.linear_velocity, "    angular velocity: ", env.robot.angular_velocity)
        print("pos traj: ", env.pos_traj, "    grid traj: ", env.grid_traj)
        step()
        return

    # Spacebar
    if event.key == ' ':
        print("Robot state: ", env.robot.pos, ' ', env.robot.orn)
        print("linear velocity: ", env.robot.linear_velocity, "    angular velocity: ", env.robot.angular_velocity)
        step()
        return

    if event.key == '.':
        print("Robot state: ", env.robot.pos, ' ', env.robot.orn)
        env.robot.linear_velocity = 0
        env.robot.angular_velocity = 0
        print("linear velocity: ", env.robot.linear_velocity, "    angular velocity: ", env.robot.angular_velocity)
        step()
        return

def constrain(input, low, high):
    if input < low:
      input = low
    elif input > high:
      input = high
    else:
      input = input

    return input

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

# test function
def detect_a_new_frontier_and_move_to_it():
    window = Window('map')
    window.show(block=False)
    # window.show_img(raw_map)
    env = GazeboEnv("/home/nics/catkin_ws/small_room_005.pgm", 0.05, 3)
    env.reset()
    # detect frontiers
    frs, _ = env.frontiers_detection()
    # choose one randomly
    fr = [267,226]
    # Astar global planning
    global_plan = env.Astar_global_planner(env.robot.pos, env.discrete_to_continuous(fr)) 
    #pose = env.naive_local_planner(global_plan)
    #env.build_map_given_path(pose)
    vis_map = env.plot_path(global_plan, env.built_map)
    window.show_img(vis_map)
    import pdb; pdb.set_trace()
    end = False
    print("target frontier: ", fr, ' ', env.discrete_to_continuous(fr))
    while(not end):
        optimal_u, optimal_traj = env.dwa_planner(env.robot.pos, env.robot.orn, global_plan)
        env.step(optimal_u)
        print("Robot state: ", env.robot.pos, ' ', env.robot.orn)
        print("linear velocity: ", env.robot.linear_velocity, "    angular velocity: ", env.robot.angular_velocity)
        end = env.arrival_detection(fr)
        vis_map = env.visualize_path()
        vis_map[fr[0],fr[1]] = 128
        vis_map = env.plot_path(global_plan, vis_map)
        window.show_img(vis_map)
    img = Image.fromarray(vis_map.astype('uint8'))
    img.show()

def use_rrt_to_explore():
    window_path = Window('path')
    window_path.show(block=False)
    window_rrt = Window('rrt')
    window_rrt.show(block=False)

    # window.show_img(raw_map)
    env = GazeboEnv("/home/nics/catkin_ws/small_room_005.pgm", 0.05, 3)
    env.reset()
    env.rrt_explore_single_thread(window_path=window_path, window_rrt=window_rrt)

def use_mmpf_to_explore():
    # window.show_img(raw_map)
    env = GazeboEnv(0.1, 3.5, 1, 1000, visualization=True)
    env.reset_for_traditional()
    while(True):
        env.step_for_mmpf()

def use_cost_method_to_explore():
    env = GazeboEnv(0.1, 3.5, 1, 1000, visualization=True)
    env.reset_for_traditional()
    while(True):
        env.step_for_cost()

def use_utility_method_to_eplore():
    window_path = Window('path')
    window_path.show(block=False)
    window_frontiers = Window('frontiers')
    window_frontiers.show(block=False)

    # window.show_img(raw_map)
    env = GazeboEnv(0.05, 3, 1, 100)
    env.reset()
    # obs, rewards, dones, infos = env.step(1)
    # import pdb; pdb.set_trace()
    env.utility_based_explore(window_path=window_path, window_frontiers=window_frontiers)

# use_rrt_to_explore()
# use_mmpf_to_eplore()
# use_cost_method_to_explore()
# use_mmpf_to_explore()
# use_cost_method_to_explore()
# # detect_a_new_frontier_and_move_to_it()
# # env = GazeboEnv("/home/nics/catkin_ws/small_room_005.pgm", 0.05, 3)
# # test = np.zeros((8,8))
# # env.simulate_laser(0,0,test)
# import pdb; pdb.set_trace()


# BURGER_MAX_LIN_VEL = 0.22
# BURGER_MAX_ANG_VEL = 2.84

# WAFFLE_MAX_LIN_VEL = 0.26
# WAFFLE_MAX_ANG_VEL = 1.82

# LIN_VEL_STEP_SIZE = 0.01
# ANG_VEL_STEP_SIZE = 0.1

# env = GazeboEnv("/home/nics/catkin_ws/small_room_005.pgm", 0.05, 3)
# env.reset()
# import pdb; pdb.set_trace()
# vis_map = env.visualize_frontiers()
# img = Image.fromarray(vis_map.astype('uint8'))
# img.show()
# global_path = env.Astar_global_planner(env.robot.pos, [1.8,-4])
# import pdb; pdb.set_trace()
# while(True):
#     optimal_u, optimal_traj = env.dwa_planner(env.robot.pos, env.robot.orn, global_path)
#     env.step(optimal_u)
#     env.visualize_path()
#     print(optimal_u)
#     print(env.robot.pos)
#     print(env.robot.orn)
#     print(env.continuous_to_discrete(env.robot.pos))
#     import pdb; pdb.set_trace()
# env.compute_robot_traj(2, 0.2)
# import pdb; pdb.set_trace()

# window = Window('map')
# window.reg_key_handler(key_handler)
# reset()
# window.show(block=True)

# for i in range(10):
#     step("right")

# window.show(block=True)
# raw_map = Image.open("/home/nics/catkin_ws/small_room.pgm")
# raw_map = np.array(raw_map)
#start_time = time.time()
# part_map, scale_map = build_map_per_frame([20, 40], 0, raw_map, 0.1, 3)
# part_map_2, scale_map_2 = build_map_per_frame([20, 21], 0, raw_map, 0.1, 3)
# part_map_3, scale_map_3 = build_map_per_frame([20, 22], 0, raw_map, 0.1, 3)
# #print("build map costs: ", time.time()-start_time)
# img = Image.fromarray(scale_map.astype('uint8'))
# img.show()
# img.save("scale_map.pgm")
# img = Image.fromarray(scale_map_2.astype('uint8'))
# img.save("scale_map_2.pgm")
# img = Image.fromarray(scale_map_3.astype('uint8'))
# img.save("scale_map_3.pgm")

# merge_map_1 = merge_two_map(scale_map, scale_map_2)
# merge_map_2 = merge_two_map(merge_map_1, scale_map_3)
# img = Image.fromarray(merge_map_1.astype('uint8'))
# img.save("merge_map_1.pgm")
# img = Image.fromarray(merge_map_2.astype('uint8'))
# img.save("merge_map_2.pgm")

# test build_map_along_traj
# traj = []
# for i in range(20):
#     traj.append([20-i, 20])

# visualize_path(traj, raw_map, "gt_map_traj_2.pgm")
# window = Window('map')
# window.show_img(raw_map)
# window.show(block=True)

# traj_map = build_map_along_traj(traj, raw_map, 0.1, 3)
# visualize_path(traj, traj_map, "traj_map_traj_2.pgm")
# img = Image.fromarray(traj_map.astype('uint8'))
# img.save("traj_map_2.pgm")
# img.show()

    
 
