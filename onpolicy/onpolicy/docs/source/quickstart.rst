Quickstart: 
=========================================


Training the Highway Environment with MAPPO Algorithm
--------------------

One line to start training the agent in Highway Environment with MAPPO Algorithm:
1. change the root path into `scripts/train/`
2. run the following code

.. code-block:: bash

    ./train_highway.sh



Hyperparameters
--------------------

Within `train_highway.sh` file, `train_highway.py` under `scripts/train/` is called with part hyperparameters specified.

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 python train/train_highway.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --task_type ${task} --n_attackers ${n_attackers} --n_defenders ${n_defenders} --n_dummies ${n_dummies} --seed ${seed} --n_training_threads 2 --n_rollout_threads 2 --n_eval_rollout_threads 2 --horizon 40 --episode_length 40 --log_interval 1 --use_wandb


Hyperparameters contain two types

- common hyperparameters used in all environments. Such parameters are parserd in ./config.py
- private hyperparameters used in specified environment itself, which are parsered by ./scripts/train/train_<env>.python

Take highway env as an example,
- the common hyperparameters are following:

.. automodule:: config
    :members:
 
- the private hyperparameters are following:

Take highway environment as an example:

.. automodule:: scripts.train.train_highway.make_train_env
    :members:

 Hierarchical structure 
 --------------------