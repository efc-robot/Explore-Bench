import numpy as np
import torch
from onpolicy.algorithms.r_mappo_single.algorithm.r_actor_critic import R_Model
from onpolicy.utils.util import update_linear_schedule


class R_MAPPOPolicy:
    def __init__(self, args, obs_space, share_obs_space, action_space, device=torch.device("cpu")):

        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.act_space = action_space

        self.model = R_Model(args, self.obs_space, self.share_obs_space, self.act_space, self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.optimizer, episode, episodes, self.lr)

    def get_actions(self, share_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None, deterministic=False):
        actions, action_log_probs, rnn_states_actor = self.model.get_actions(
            obs, rnn_states_actor, masks, available_actions, deterministic)
        values, rnn_states_critic = self.model.get_values(
            share_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, share_obs, rnn_states_critic, masks):
        values, _ = self.model.get_values(share_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, obs, rnn_states_actor, action, masks, available_actions=None, active_masks=None):
        action_log_probs, dist_entropy = self.model.evaluate_actions(
            obs, rnn_states_actor, action, masks, available_actions, active_masks)

        return action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        actions, _, rnn_states_actor = self.model.get_actions(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
