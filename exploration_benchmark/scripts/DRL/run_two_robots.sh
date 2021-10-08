#!/bin/sh
env="Gazebo"
num_agents=2
algo="mappo"
exp1="robot1"
exp2="robot2"

echo "env is ${env}, algo is ${algo}, exp is ${exp1}"
CUDA_VISIBLE_DEVICES=0 python two_robot_1.py \
--env_name ${env} --algorithm_name ${algo} --experiment_name ${exp1} \
--user_name "xyf" --num_agents ${num_agents} \
--cnn_layers_params '16,3,1,1 32,3,1,1 16,3,1,1' --hidden_size 64 \
--use_merge --use_recurrent_policy &

echo "env is ${env}, algo is ${algo}, exp is ${exp2}"
CUDA_VISIBLE_DEVICES=0 python two_robot_2.py \
--env_name ${env} --algorithm_name ${algo} --experiment_name ${exp2} \
--user_name "xyf" --num_agents ${num_agents} \
--cnn_layers_params '16,3,1,1 32,3,1,1 16,3,1,1' --hidden_size 64 \
--use_merge --use_recurrent_policy &
