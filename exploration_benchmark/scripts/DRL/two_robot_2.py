import sys
import time
import os
import numpy as np
import setproctitle
import torch
import torch.nn as nn
import gym
from PIL import Image
import pickle
import copy
import random
import math
from window import Window
import rospy
import actionlib
from std_msgs.msg import String, Float32MultiArray
from nav_msgs.msg import OccupancyGrid, Odometry
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
from onpolicy.config import get_config

import cv2

def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str, default='simple_spread', help="Which scenario to run on")
    parser.add_argument('--num_agents', type=int, default=2, help="number of players")
    parser.add_argument('--num_obstacles', type=int, default=1, help="number of players")
    parser.add_argument('--agent_pos', type=list, default = None, help="agent_pos")
    parser.add_argument('--grid_size', type=int, default=19, help="map size")
    parser.add_argument('--agent_view_size', type=int, default=7, help="depth the agent can view")
    parser.add_argument('--max_steps', type=int, default=100, help="depth the agent can view")
    parser.add_argument('--local_step_num', type=int, default=3, help="local_goal_step")
    parser.add_argument("--use_same_location", action='store_true', default=False,
                        help="use merge information")
    parser.add_argument("--use_single_reward", action='store_true', default=False,
                        help="use single reward")
    parser.add_argument("--use_complete_reward", action='store_true', default=False,
                        help="use complete reward")            
    parser.add_argument("--use_merge", action='store_true', default=False,
                        help="use merge information")
    parser.add_argument("--use_multiroom", action='store_true', default=False,
                        help="use multiroom")
    parser.add_argument("--use_random_pos", action='store_true', default=False,
                        help="use complete reward")   
    parser.add_argument("--use_time_penalty", action='store_true', default=False,
                        help="use time penalty")    
    parser.add_argument("--use_intrinsic_reward", action='store_true', default=False,
                        help="use intrinsic reward")             
    parser.add_argument("--visualize_input", action='store_true', default=False,
                        help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    all_args = parser.parse_known_args(args)[0]

    return all_args

class RLRunner(object):
    def __init__(self, args, device, model_dir=None):
        self.args = args
        self.device = device
        self.model_dir = model_dir

        self.resize_width = 64
        self.resize_height = 64

        self.num_agents = 1
        self.use_centralized_V = self.args.use_centralized_V
        use_merge = True
        self.use_merge = use_merge

        self.num_agents = 1
        self.augment = 255 // (np.array([agent_id+1 for agent_id in range(self.num_agents)]).sum())

        # self.tf_listener = tf.TransformListener()
        self.robot_pose = None
        self.path = []

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
        global_observation_space['vector'] = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype='float')
        if use_merge:
            global_observation_space['global_merge_obs'] = gym.spaces.Box(
                low=0, high=255, shape=(4, self.resize_width, self.resize_height), dtype='uint8')
        #     global_observation_space['global_direction'] = gym.spaces.Box(
        #         low=-1, high=1, shape=(self.num_agents, 4), dtype='float')
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

        share_observation_space = self.share_observation_space[0] if self.use_centralized_V else self.observation_space[0]

        # policy network
        self.policy = Policy(self.args,
                            self.observation_space[0],
                            share_observation_space,
                            self.action_space[0],
                            device = self.device)

        if self.model_dir is not None:
            self.restore()

        # algorithm
        self.trainer = TrainAlgo(self.args, self.policy, device = self.device)

        # visualization
        self.window = Window('map')
        self.window.show(block=False)

        # begin ROS spin
        self.listener()
    
    def restore(self):
        print("rostore model successfully")
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt', map_location=self.device)
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.args.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt', map_location=self.device)
            self.policy.critic.load_state_dict(policy_critic_state_dict)

    def init_eval_map_variables(self):
        # Initializing full, merge and local map
        self.eval_all_agent_pos_map = np.zeros((1, self.num_agents, self.full_h, self.full_w), dtype=np.float32)
        if self.use_merge:
            #self.eval_global_merge_goal_trace = np.zeros((self.n_eval_rollout_threads, self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
            self.eval_all_merge_pos_map = np.zeros((1, self.full_h, self.full_w), dtype=np.float32)

    def resize_obs(self, infos):
        self.init_eval_map_variables()
        obs = {}
        # obs['vector'] = np.zeros((1, self.num_agents, self.num_agents), dtype=np.float32)
        obs['vector'] = np.zeros((1, self.num_agents, 2), dtype=np.float32)
        obs['global_obs'] = np.zeros((1, self.num_agents, 4, self.resize_width, self.resize_height), dtype=np.float32)
        agent_pos_map = np.zeros((1, self.num_agents, self.full_h, self.full_w), dtype=np.float32)
        if self.use_merge:
            obs['global_merge_obs'] = np.zeros((1, self.num_agents, 4, self.resize_width, self.resize_height), dtype=np.float32)
            merge_pos_map = np.zeros((1, self.full_h, self.full_w), dtype=np.float32)
        for agent_id in range(self.num_agents):
            agent_pos_map[0 , agent_id, infos['current_agent_pos'][agent_id][0], infos['current_agent_pos'][agent_id][1]] = (agent_id + 1) * self.augment
            self.eval_all_agent_pos_map[0, agent_id, infos['current_agent_pos'][agent_id][0], infos['current_agent_pos'][agent_id][1]] = (agent_id + 1) * self.augment
            if self.use_merge:
                merge_pos_map[0, infos['current_agent_pos'][agent_id][0], infos['current_agent_pos'][agent_id][1]] += (agent_id + 1) * self.augment
                if self.eval_all_merge_pos_map[0, infos['current_agent_pos'][agent_id][0], infos['current_agent_pos'][agent_id][1]] != (agent_id + 1) * self.augment and\
                    self.eval_all_merge_pos_map[0, infos['current_agent_pos'][agent_id][0], infos['current_agent_pos'][agent_id][1]] != np.array([agent_id + 1 for agent_id in range(self.num_agents)]).sum() * self.augment:
                        self.eval_all_merge_pos_map[0, infos['current_agent_pos'][agent_id][0], infos['current_agent_pos'][agent_id][1]] += (agent_id + 1) * self.augment   
        for agent_id in range(self.num_agents):
            obs['global_obs'][0, agent_id, 0] = cv2.resize(infos['explored_each_map'][agent_id].astype(np.uint8), (self.resize_width, self.resize_height)).astype(np.float32)
            obs['global_obs'][0, agent_id, 1] = cv2.resize(infos['obstacle_each_map'][agent_id].astype(np.uint8), (self.resize_width, self.resize_height)).astype(np.float32)
            obs['global_obs'][0, agent_id, 2] = cv2.resize((agent_pos_map[0, agent_id] > 0).astype(np.uint8), (self.resize_width, self.resize_height)).astype(np.float32)
            obs['global_obs'][0, agent_id, 3] = cv2.resize((self.eval_all_agent_pos_map[0, agent_id] > 0).astype(np.uint8), (self.resize_width, self.resize_height)).astype(np.float32)
            if self.use_merge:
                obs['global_merge_obs'][0, agent_id, 0] = cv2.resize(infos['explored_all_map'].astype(np.uint8), (self.resize_width, self.resize_height)).astype(np.float32)
                obs['global_merge_obs'][0, agent_id, 1] = cv2.resize(infos['obstacle_all_map'].astype(np.uint8), (self.resize_width, self.resize_height)).astype(np.float32)
                obs['global_merge_obs'][0, agent_id, 2] = cv2.resize((merge_pos_map[0] > 0).astype(np.uint8), (self.resize_width, self.resize_height)).astype(np.float32)
                obs['global_merge_obs'][0, agent_id, 3] = cv2.resize((self.eval_all_merge_pos_map[0] > 0).astype(np.uint8), (self.resize_width, self.resize_height)).astype(np.float32)

            obs['vector'][0, agent_id] = np.eye(2)[0] 
        return obs            
    
    def merge_map_callback(self, data):
        self.merge_map = data

    def map_callback(self, data):
        # -1:unkown 0:free 100:obstacle   
        # ------------- maybe differ in map type -----------------
        # print("receive map")
        self.map = data
    
    def costmap_callback(self, data):
        # print('receive costmap')
        self.costmap = data

    def odom_callback(self, data):
        # print("receive odom")
        self.robot_pose = [data.pose.pose.position.x, data.pose.pose.position.y]
    
    def tf_callback(self, data):
        # print("receive odom")
        self.robot_pose = [data.data[0], data.data[1]]

    def output_goal(self, temp_map, temp_costmap, temp_pose, temp_merge_map):
        self.full_w, self.full_h = temp_merge_map.info.width, temp_merge_map.info.height
        # print(self.full_w, ' ', self.full_h)
        # -1:unkown 0:free 100:obstacle 
        self_gridmap = np.array(temp_map.data).reshape((temp_map.info.height, temp_map.info.width))
        self_gridmap[self_gridmap>50] = -100
        self_gridmap[self_gridmap>0] = 0
        self_gridmap[self_gridmap<-1] = 100
        merge_gridmap = np.array(temp_merge_map.data).reshape((temp_merge_map.info.height, temp_merge_map.info.width))
        # debug 
        delta_x = int((temp_map.info.origin.position.x - temp_merge_map.info.origin.position.x)/temp_map.info.resolution)
        delta_y = int((temp_map.info.origin.position.y - temp_merge_map.info.origin.position.y)/temp_map.info.resolution)
        gridmap = np.zeros((temp_merge_map.info.height, temp_merge_map.info.width))
        gridmap[:] = -1

        # can be modified
        if delta_y >=0 and delta_y+temp_map.info.height <= temp_merge_map.info.height and delta_x >=0 and delta_x+temp_map.info.width <= temp_merge_map.info.width:
            gridmap[delta_y:delta_y+temp_map.info.height, delta_x:delta_x+temp_map.info.width] = self_gridmap
        else:
            print("maps differ in size, so use the merge_map")
            gridmap = copy.deepcopy(merge_gridmap)
        
        costmap = np.array(temp_costmap.data).reshape((temp_costmap.info.height, temp_costmap.info.width))
    

        # cutting
        # gridmap = gridmap[:300,:200]
        # costmap = gridmap[:300,:200]
        # self.full_h, self.full_w = gridmap.shape[0], gridmap.shape[1]

        RobotData = {}
        RobotData['OccupancyGrid'] = gridmap
        RobotData['Merge_Map'] = merge_gridmap
        RobotData['costmap'] = costmap

        info = {}
        agent_pos = [0, 0]
        agent_pos[0] = math.floor((temp_pose[1] - temp_merge_map.info.origin.position.y)/temp_merge_map.info.resolution)
        agent_pos[1] = math.floor((temp_pose[0] - temp_merge_map.info.origin.position.x)/temp_merge_map.info.resolution)
        print(agent_pos)
        self.path.append(agent_pos)
        explored_all_map = (merge_gridmap != -1).astype(int)
        obstacle_all_map = (merge_gridmap == 100).astype(int)

        explored_single_map = (gridmap != -1).astype(int)
        obstacle_single_map = (gridmap == 100).astype(int)

        explored_each_map = []
        obstacle_each_map = []
        current_agent_pos = []
        current_agent_dir = []
        explored_each_map.append(explored_single_map)
        obstacle_each_map.append(obstacle_single_map) 
        current_agent_pos.append(agent_pos)
        current_agent_dir.append(random.randint(0, 3))
        info['explored_all_map'] = np.array(explored_all_map)
        info['current_agent_pos'] = np.array(current_agent_pos)
        info['explored_each_map'] = np.array(explored_each_map)
        info['obstacle_all_map'] = np.array(obstacle_all_map)
        info['obstacle_each_map'] = np.array(obstacle_each_map)
        info['agent_direction'] = np.array(current_agent_dir)
        obs = self.resize_obs(info)
        eval_rnn_state = np.zeros((self.num_agents, self.args.recurrent_N, self.args.hidden_size), dtype=np.float32)
        eval_mask = np.ones((self.num_agents, 1), dtype=np.float32)
        self.trainer.prep_rollout()
        concat_eval_obs = {}
        for key in obs.keys():
            concat_eval_obs[key] = obs[key][0]
        # import pickle
        #     # output_file = open('gazebo.pkl', 'wb')
        #     # pickle.dump(concat_eval_obs, output_file)
        # input_file = open('testbed.pkl', 'rb')
        # concat_eval_obs_load = pickle.load(input_file)

        eval_action, eval_rnn_state = self.trainer.policy.act(concat_eval_obs,
                                        eval_rnn_state,
                                        eval_mask,
                                        deterministic=True)
        # import pickle
        # output_file = open('gazebo.pkl', 'wb')
        # pickle.dump(concat_eval_obs, output_file)
        RobotData['global_goal'] = nn.Sigmoid()(eval_action)[0]
        print('RL output: ', RobotData['global_goal'])
        RobotData['width'] = temp_merge_map.info.width
        RobotData['height'] = temp_merge_map.info.height
        RobotData['pose'] = temp_pose
        # if too close, choose another one
        goal, raw_goal = self.get_short_term_goal(RobotData)

        self.goal.target_pose.header.frame_id ='robot2/map'
        self.goal.target_pose.header.stamp = rospy.Time.now()
        self.goal.target_pose.pose.position.x = temp_merge_map.info.origin.position.x + goal[1]*temp_merge_map.info.resolution
        self.goal.target_pose.pose.position.y = temp_merge_map.info.origin.position.y + goal[0]*temp_merge_map.info.resolution
        self.goal.target_pose.pose.orientation.w = 1.0
        print(self.goal.target_pose.pose.position.x, ' ', self.goal.target_pose.pose.position.y)

        # visualization
        vis_map = copy.deepcopy(gridmap)
        vis_map[gridmap==0] = 30
        vis_map[gridmap==-1] = 0
        if raw_goal[0] > 0 and raw_goal[0] < vis_map.shape[0]-1 and raw_goal[1] > 0 and raw_goal[1] < vis_map.shape[1]-1:
            vis_map[raw_goal[0],raw_goal[1]] = 80
            vis_map[raw_goal[0]-1,raw_goal[1]] = 80
            vis_map[raw_goal[0]+1,raw_goal[1]] = 80
            vis_map[raw_goal[0],raw_goal[1]-1] = 80
            vis_map[raw_goal[0]-1,raw_goal[1]-1] = 80
            vis_map[raw_goal[0]+1,raw_goal[1]-1] = 80
            vis_map[raw_goal[0],raw_goal[1]+1] = 80
            vis_map[raw_goal[0]-1,raw_goal[1]+1] = 80
            vis_map[raw_goal[0]+1,raw_goal[1]+1] = 80
        if goal[0] > 0 and goal[0] < vis_map.shape[0]-1 and goal[1] > 0 and goal[1] < vis_map.shape[1]-1:
            vis_map[goal[0],goal[1]] = 50
            vis_map[goal[0]-1,goal[1]] = 50
            vis_map[goal[0]+1,goal[1]] = 50
            vis_map[goal[0],goal[1]-1] = 50
            vis_map[goal[0]-1,goal[1]-1] = 50
            vis_map[goal[0]+1,goal[1]-1] = 50
            vis_map[goal[0],goal[1]+1] = 50
            vis_map[goal[0]-1,goal[1]+1] = 50
            vis_map[goal[0]+1,goal[1]+1] = 50
        # self.window.show_img(vis_map) 
        self.client.send_goal(self.goal)

        # Waits for the server to finish performing the action.
        print("Send the goal and move to it.....")
        self.client.wait_for_result(rospy.Duration.from_sec(8.0))
        print("Finish one goal") 

    def get_short_term_goal(self, data):
        occupancy_grid = data['OccupancyGrid']
        goal = [int(data['height']*data['global_goal'][0]), int(data['width']*data['global_goal'][1])]
        frs, _ = self.detect_frontiers(occupancy_grid, data['costmap'])
        # cluster targets into different groups and find the center of each group.
        target_process = copy.deepcopy(frs)
        cluster_center = []
        infoGain_cluster = []
        while(len(target_process) > 0):
            target_cluster = []
            target_cluster.append(target_process.pop())

            condition = True
            while(condition):
                condition = False
                size_target_process = len(target_process)
                for i in reversed(range(size_target_process)):
                    for j in range(len(target_cluster)):
                        dis = abs(target_process[i][0] - target_cluster[j][0]) + abs(target_process[i][1] - target_cluster[j][1])
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

        if len(cluster_center) > 0:
            min_dis = 10000
            min_idx = -1
            for idx, fr in enumerate(cluster_center):
                dis = math.sqrt(math.hypot(fr[0]-goal[0], fr[1]-goal[1]))
                if dis < min_dis:
                    min_dis = dis
                    min_idx = idx
            # map_goal.append(free_cluster_center[min_idx])
            map_goal = cluster_center[min_idx]
        else:
            map_goal = data['pose']
        return map_goal, goal

    def detect_frontiers(self, explored_map, costmap):
        obstacles = []
        frontiers = []
        height = explored_map.shape[0]
        width = explored_map.shape[1]
        for i in range(2, height-2):
            for j in range(2, width-2):
                if explored_map[i][j] == 100:
                    obstacles.append([i,j])
                elif explored_map[i][j] == -1:
                    numFree = 0
                    temp1 = 0
                    if explored_map[i+1][j] == 0:
                        temp1 += 1 if explored_map[i+2][j] == 0 else 0
                        temp1 += 1 if explored_map[i+1][j+1] == 0 else 0
                        temp1 += 1 if explored_map[i+1][j-1] == 0 else 0
                        numFree += (temp1 > 0)
                    if explored_map[i][j+1] == 0:
                        temp1 += 1 if explored_map[i][j+2] == 0 else 0
                        temp1 += 1 if explored_map[i+1][j+1] == 0 else 0
                        temp1 += 1 if explored_map[i-1][j+1] == 0 else 0
                        numFree += (temp1 > 0)     
                    if explored_map[i-1][j] == 0:
                        temp1 += 1 if explored_map[i-1][j+1] == 0 else 0
                        temp1 += 1 if explored_map[i-1][j-1] == 0 else 0
                        temp1 += 1 if explored_map[i-2][j] == 0 else 0
                        numFree += (temp1 > 0)
                    if explored_map[i][j-1] == 0:
                        temp1 += 1 if explored_map[i][j-2] == 0 else 0
                        temp1 += 1 if explored_map[i+1][j-1] == 0 else 0
                        temp1 += 1 if explored_map[i-1][j-1] == 0 else 0
                        numFree += (temp1 > 0)     
                    if numFree > 0:
                        frontiers.append([i,j])  
        valid_frs = []
        for fr in frontiers:
            if costmap[fr[0], fr[1]] <= 0:
                valid_frs.append(fr)
        return valid_frs, obstacles

    def listener(self):
        rospy.init_node('robot2_save_2D_map', anonymous=True)
        self.client = actionlib.SimpleActionClient('robot2/move_base', MoveBaseAction)
        self.client.wait_for_server()

        self.goal = MoveBaseGoal()
        self.map = OccupancyGrid()
        self.costmap = OccupancyGrid()
        self.merge_map = OccupancyGrid()
        self.robot_pose = []
        rospy.Subscriber("/robot2/cartographer_discrete_map", OccupancyGrid, self.map_callback)
        rospy.Subscriber("/robot2/move_base/global_costmap/costmap", OccupancyGrid, self.costmap_callback)
        rospy.Subscriber("/robot2/map", OccupancyGrid, self.merge_map_callback)
        # rospy.Subscriber("/robot1/odom", Odometry, self.odom_callback)
        # for cartographer
        rospy.Subscriber("/robot2/robot_pos", Float32MultiArray, self.tf_callback)
        rate = rospy.Rate(10)

        while(len(self.costmap.data) < 1):
            # print("no costmap")
            pass
        while(len(self.map.data) < 1):
            # print("no map")
            pass
        while(len(self.merge_map.data) < 1):
            pass
        while(len(self.robot_pose) < 1):
            # print("no robot_pose")
            pass
        while not rospy.is_shutdown():
            print("map time stamp: ", self.map.header.stamp)
            print("costmap time stamp: ", self.costmap.header.stamp)
            temp_map, temp_costmap, temp_pose, temp_merge_map = self.map, self.costmap, self.robot_pose, self.merge_map
            self.output_goal(temp_map, temp_costmap, temp_pose, temp_merge_map)
            rate.sleep()

if __name__ == '__main__':
    parser = get_config()
    all_args = parse_args(sys.argv[1:], parser)
    if all_args.algorithm_name == "rmappo" or all_args.algorithm_name == "rmappg":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mappg":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), ("check recurrent policy!")
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    runner = RLRunner(all_args, device, "./models")
