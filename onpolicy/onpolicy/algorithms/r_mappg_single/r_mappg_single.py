import numpy as np
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check

class R_MAPPG():
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.aux_epoch = args.aux_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.clone_coef = args.clone_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks

        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device = self.device)
        else:
            self.value_normalizer = None

    def policy_loss_update(self, sample):
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        action_log_probs, dist_entropy = self.policy.evaluate_actions(obs_batch, rnn_states_batch, actions_batch, masks_batch, available_actions_batch, active_masks_batch)

        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        action_loss = (-torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()

        # update common and action network
        self.policy.optimizer.zero_grad()
        (action_loss - dist_entropy * self.entropy_coef).backward()
        if self._use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.model.parameters(), self.max_grad_norm)
        else:
            grad_norm = get_gard_norm(self.policy.model.parameters())
        self.policy.optimizer.step()

        return action_loss, dist_entropy, grad_norm, ratio

    def value_loss_update(self, sample):
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch = sample

        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # freeze common network
        for p in self.policy.model.common.parameters():
            p.requires_grad = False
        if self._use_recurrent_policy or self._use_naive_recurrent_policy:
            for p in self.policy.model.rnn.parameters():
                p.requires_grad = False

        values = self.policy.get_values(share_obs_batch, rnn_states_critic_batch, masks_batch)

        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
            
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() /  active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        self.policy.optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.max_grad_norm)
        else:
            grad_norm = get_gard_norm(self.policy.model.parameters())

        self.policy.optimizer.step()

        return value_loss, grad_norm

    def update_action_probs(self, buffer):
        action_probs = []
        for step in range(buffer.episode_length):
            if buffer.available_actions is not None:
                avail_actions = np.concatenate(buffer.available_actions[step])
            else:
                avail_actions = None
            action_prob = self.policy.get_probs(np.concatenate(buffer.obs[step]),
                                                np.concatenate(buffer.rnn_states[step]),
                                                np.concatenate(buffer.masks[step]),
                                                avail_actions)
            action_prob = np.array(np.split(action_prob.detach().cpu().numpy(), buffer.n_rollout_threads))
            action_probs.append(action_prob)
            
        return np.array(action_probs)

    def auxiliary_loss_update(self, sample):
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            old_action_probs_batch, available_actions_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        old_action_probs_batch = check(old_action_probs_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, new_action_probs = self.policy.get_values_and_probs(
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, masks_batch, available_actions_batch)

        # kl = sum p * log(p / q) = sum p*(logp-logq) = sum plogp - plogq
        # cross-entropy = sum -plogq 
        kl_divergence = torch.sum((old_action_probs_batch * (old_action_probs_batch.log()-new_action_probs.log())), dim=-1, keepdim=True)
        kl_loss = (kl_divergence * active_masks_batch).sum() / active_masks_batch.sum()
        
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
            
        if self._use_popart or self._use_valuenorm:
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        joint_loss = value_loss + self.clone_coef * kl_loss

        self.policy.optimizer.zero_grad()

        joint_loss.backward()

        if self._use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.max_grad_norm)
        else:
            grad_norm = get_gard_norm(self.policy.model.parameters())

        self.policy.optimizer.step()

        return joint_loss, grad_norm

    def train(self, buffer):
        if self._use_popart:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['action_loss'] = 0
        train_info['joint_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['joint_grad_norm'] = 0
        train_info['ratio'] = 0

        # policy phase
        for _ in range(self.ppo_epoch):

            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent_policy:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                action_loss, dist_entropy, actor_grad_norm, ratio = self.policy_loss_update(sample)

                train_info['action_loss'] += action_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm.item()
                train_info['ratio'] += ratio.mean().item() 

                value_loss, critic_grad_norm = self.value_loss_update(sample)

                train_info['value_loss'] += value_loss.item()
                train_info['critic_grad_norm'] += critic_grad_norm.item()

        # auxiliary phase
        action_probs = self.update_action_probs(buffer)

        for _ in range(self.aux_epoch):

            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(action_probs, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent_policy:
                data_generator = buffer.naive_recurrent_generator(action_probs, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(action_probs, self.num_mini_batch)

            # 2. update auxiliary
            for sample in data_generator:

                joint_loss, joint_grad_norm = self.auxiliary_loss_update(sample)

                train_info['joint_loss'] += joint_loss.item()
                train_info['joint_grad_norm'] += joint_grad_norm.item()

        for k in train_info.keys():
            if k in ["joint_loss","joint_grad_norm"]:
                num_updates = self.aux_epoch * self.num_mini_batch
                train_info[k] /= num_updates
            else:
                num_updates = self.ppo_epoch * self.num_mini_batch
                train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        self.policy.model.train()

    def prep_rollout(self):
        self.policy.model.eval()
