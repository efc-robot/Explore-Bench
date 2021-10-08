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

class GridEnv(gym.Env):
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
        map_file = random.choice(os.listdir('/home/nics/git_ws/src/onpolicy/onpolicy/envs/GridEnv/datasets'))
        map_img = Image.open(os.path.join('/home/nics/git_ws/src/onpolicy/onpolicy/envs/GridEnv/datasets', map_file))
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

        if self.visualization:
            self.visualize_map = copy.deepcopy(self.complete_map)
            for pt in self.visualize_goal:
                if pt[0] > 0 and pt[0] < 299 and pt[1] > 0 and pt[1] < 299:
                    self.visualize_map[pt[0], pt[1]] = 128
                    self.visualize_map[pt[0]-1, pt[1]] = 128
                    self.visualize_map[pt[0]+1, pt[1]] = 128
                    self.visualize_map[pt[0], pt[1]-1] = 128
                    self.visualize_map[pt[0]-1, pt[1]-1] = 128
                    self.visualize_map[pt[0]+1, pt[1]-1] = 128
                    self.visualize_map[pt[0], pt[1]+1] = 128
                    self.visualize_map[pt[0]-1, pt[1]+1] = 128
                    self.visualize_map[pt[0]+1, pt[1]+1] = 128
                else:
                    self.visualize_map[pt[0], pt[1]] = 128

            self.window.show_img(self.visualize_map)
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

    def continuous_to_discrete(self, pos):
        idx_x = int(pos[0] / self.resolution) + int(self.gt_map.shape[0]/2)
        idx_y = int(pos[1] / self.resolution) + int(self.gt_map.shape[1]/2)
        return [idx_x, idx_y]

    def discrete_to_continuous(self, grid_idx):
        pos_x = (grid_idx[0] - self.gt_map.shape[0]/2) * self.resolution
        pos_y = (grid_idx[1] - self.gt_map.shape[1]/2) * self.resolution
        return [pos_x, pos_y]

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
    env = GridEnv("/home/nics/catkin_ws/small_room_005.pgm", 0.05, 3)
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
    env = GridEnv("/home/nics/catkin_ws/small_room_005.pgm", 0.05, 3)
    env.reset()
    env.rrt_explore_single_thread(window_path=window_path, window_rrt=window_rrt)

def use_mmpf_to_explore():
    # window.show_img(raw_map)
    env = GridEnv(0.1, 3.5, 1, 1000, visualization=True)
    env.reset_for_traditional()
    while(True):
        env.step_for_mmpf()

def use_cost_method_to_explore():
    env = GridEnv(0.1, 3.5, 1, 1000, visualization=True)
    env.reset_for_traditional()
    while(True):
        env.step_for_cost()

def use_utility_method_to_eplore():
    window_path = Window('path')
    window_path.show(block=False)
    window_frontiers = Window('frontiers')
    window_frontiers.show(block=False)

    # window.show_img(raw_map)
    env = GridEnv(0.05, 3, 1, 100)
    env.reset()
    # obs, rewards, dones, infos = env.step(1)
    # import pdb; pdb.set_trace()
    env.utility_based_explore(window_path=window_path, window_frontiers=window_frontiers)
    
 
