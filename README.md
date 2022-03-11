# Explore-Bench

**Explore-Bench** is developed aiming to evaluate traditional frontier-based and deep-reinforcement-learning-based autonomous exploration approaches in a unified and comprehensive way.  

The related paper ["Explore-Bench: Data Sets, Metrics and Evaluations for Frontier-based and Deep-reinforcement-learning-based Autonomous Exploration"](https://arxiv.org/abs/2202.11931) is accepted to the 2022 International Conference on Robotics and Automation (ICRA 2022).

Features:

- *Data Sets*: various basic exploration scenarios (i.e., *loop, narrow corridor, corner,* and *multiple rooms*) and their combinations are designed.
- *Metrics:* two types of metrics (**efficiency** metrics and **collaboration** metrics) are proposed.
- *Platform*: a 3-level platform with a unified data flow and $12 \times$ speed-up that includes a grid-based simulator for fast evaluation and efficient training, a realistic Gazebo simulator, and a remotely accessible robot testbed for high-accuracy tests in physical environments is built.
- *Evaluations:* one DRL-based and three frontier-based exploration approaches are evaluated and some insights about the selection and design of exploration methods are provided.

## Dependency

The project has been tested on Ubuntu 18.04 (ROS Melodic). To run the exploration approach on Desktop PC or real robots, please install these packages first:

### Cartographer for map building

Cartographer is a 2D/3D map-building method. It provides the submaps' and the trajectories' information when building the map. We slightly modified the original Cartographer to make it applicable to multi-robot SLAM and exploration. Please refer to [Cartographer-for-SMMR](https://github.com/efc-robot/Cartographer-for-SMMR) to install the modified Cartographer to `carto_catkin_ws` and

```
source /PATH/TO/CARTO_CATKIN_WS/devel_isolated/setup.bash
```

### MAPPO for reinforcement learning training and evaluation

MAPPO is a multi-agent variant of PPO (Proximal Policy Optimization), which is a SOTA on-policy reinforcement learning algorithm. We slightly modified the original [marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy) and provide the training and test code in this repo.

```bash
# create conda environment
conda create -n marl python==3.8.10
conda activate marl
pip install torch torchvision
```

```bash
# install onpolicy package
cd onpolicy
pip install -e .
```

We provide requirement.txt but it may have redundancy. We recommend that the user try to install other required packages by running the code and finding which required package hasn't installed yet.

### Turtlebot3 Description and Simulation

```bash
sudo apt install ros-melodic-turtlebot3*
sudo apt install ros-melodic-bfl
pip install future
sudo apt install ros-melodic-teb-local-planner
echo 'export TURTLEBOT3_MODEL=burger' > ~/.bashrc
```

After installing these dependencies, put these packages in your ROS workspace (i.e. `catkin_ws/src`) and `catkin_make`.

## Run Frontier-based Exploration in Gazebo (Level-1)

Template:

```bash
{env}      = 'loop' or 'corridor' or 'corner' or 'room' or 'loop_with_corridor' or 'room_with_corner'
# for multiple robots, there are two env cases: 'env_close' and 'env_far' (i.e. 'loop_close' and 'loop_far')
{number_robots} = 'single' or 'two'
{method}        = 'rrt' or 'mmpf' or 'cost'
{suffix}        = 'robot' or 'robots' (be 'robot' when number_robots != 'single')
# build simulation environment
roslaunch sim_env {env}_env_{number_robots}_{suffix}.launch
# start cartographer map building and move_base
roslaunch sim_env {number_robots}_{suffix}.launch
# start frontier-based exploration
roslaunch exploration_benchmark {number_robots}_{method}_node.launch
# start data logging and evaluating the exploration performance
cd exploration_benchmark/scripts
python exploration_metric_for_{number_robots}_{suffix}.py '../blueprints/{env}.pgm' '../blueprints/{env}.yaml'
```

For example, 

- Room Environment -- Single Robot -- Field-based Exploration (MMPF)
```bash
roslaunch sim_env room_env_single_robot.launch
roslaunch sim_env single_robot.launch
roslaunch exploration_benchmark single_mmpf_node.launch
```

 Then, start a new terminal and use our proposed metrics to evaluate the exploration performance:
```bash
cd exploration_benchmark/scripts
python exploration_metric_for_single_robot.py '../blueprints/room.pgm' '../blueprints/room.yaml'
```
 At last, choose "Publish Point" button in the rviz and then click anywhere in the map to start the exploration.

- Corridor Environment (Far) -- Two Robots -- Cost-based Exploration 

```bash
roslaunch sim_env corridor_far_env_two_robots.launch
roslaunch sim_env two_robots.launch
roslaunch exploration_benchmark two_cost_node.launch
```

 Then, start a new terminal and use our proposed metrics to evaluate the exploration performance:

```bash
cd exploration_benchmark/scripts
python exploration_metric_for_two_robots.py '../blueprints/corridor.pgm' '../blueprints/corridor.yaml'
```

## Run DRL-based Exploration in Gazebo (Level-1)

To run the DRL-based exploration, we need Python3 and conda environment.

### Single Robot

First, start the simulation environment , mapping module, rviz visualization and performance evaluation,
```bash
roslaunch sim_env room_env_single_robot.launch
roslaunch sim_env single_robot.launch
roslaunch exploration_benchmark single_rl_node.launch
cd exploration_benchmark/scripts
python exploration_metric_for_single_robot.py '../blueprints/room.pgm' '../blueprints/room.yaml'
```

Then, start a new terminal and run the DRL-based exploration,
```bash
# enter conda env
source ~/anaconda3/bin/env marl
cd exploration_benchmark/scripts/DRL
# run the DRL model
bash run_single_robot.sh
```

### Multiple Robots

First, start the simulation environment , mapping module, rviz visualization and performance evaluation,

```bash
roslaunch sim_env room_far_env_two_robots.launch
roslaunch sim_env two_robots.launch
roslaunch exploration_benchmark two_rl_node.launch
cd exploration_benchmark/scripts
python exploration_metric_for_two_robots.py '../blueprints/room.pgm' '../blueprints/room.yaml'
```

Then, start a new terminal and run the DRL-based exploration,

```bash
# enter conda env
source ~/anaconda3/bin/env marl
cd exploration_benchmark/scripts/DRL
# run the DRL model
bash run_two_robots.sh
```

Note: you need to choose "Publish Point" button in the rviz and then click anywhere in the map to start the performance evaluation.

## Train DRL Model in Grid-based Simulator (Level-0)

Explore-Bench supports the DRL training in a fast grid-based simulator (Level-0).

Follow these steps to train your own DRL model:

```bash
# enter conda env
source ~/anaconda3/bin/env marl
# ensure that you have installed the MAPPO dependency
cd onpolicy/onpolicy/scripts
# run the DRL training 
bash train_grid.sh
```

The training parameters can be modified according to the user's need, i.e., the number of robots (`num_agents`), the hidden size, the batch size and so on.

Refer to `train_grid.sh` for details.

## Evaluate Exploration Approaches in Grid-based Simulator (Level-0)

Besides training DRL models, the grid-based simulator can be used for fast evaluation of both frontier-based and DRL-based methods.

The field-based and cost-based exploration approaches are taken for example:
```bash
# enter conda env
source ~/anaconda3/bin/env marl
cd grid_simulator
# evaluate cost in corner env (2 robots)
python GridEnv.py cost 2 ../onpolicy/onpolicy/envs/GridEnv/datasets/corner.pgm
# evaluate cost in corner env (1 robots)
python GridEnv.py cost 1 ../onpolicy/onpolicy/envs/GridEnv/datasets/corner.pgm
# evaluate mmpf in corner env (2 robots)
python GridEnv.py mmpf 2 ../onpolicy/onpolicy/envs/GridEnv/datasets/corner.pgm
# evaluate mmpf in corner env (1 robots)
python GridEnv.py mmpf 1 ../onpolicy/onpolicy/envs/GridEnv/datasets/corner.pgm
```
