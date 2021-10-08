import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase, MLPLayer
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.popart import PopArt

from onpolicy.utils.util import get_shape_from_obs_space

class R_Model(nn.Module):
    def __init__(self, args, obs_space, share_obs_space, action_space, device=torch.device("cpu")):
        super(R_Model, self).__init__()
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._activation_id = args.activation_id
        self._recurrent_N = args.recurrent_N
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_centralized_V = args.use_centralized_V
        self._use_popart = args.use_popart
        self.hidden_size = args.hidden_size
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        # obs space
        obs_shape = get_shape_from_obs_space(obs_space)
        self.obs_prep = CNNBase(args, obs_shape) if len(obs_shape)==3 else MLPBase(args, obs_shape, use_attn_internal=args.use_attn_internal, use_cat_self=True)
                
        # share obs space
        if self._use_centralized_V:
            share_obs_shape = get_shape_from_obs_space(share_obs_space)
            self.share_obs_prep = CNNBase(args, share_obs_shape) if len(obs_shape)==3 else MLPBase(args, share_obs_shape, use_attn_internal=True, use_cat_self=args.use_cat_self)           
        else:
            self.share_obs_prep = self.obs_prep

        # common layer
        self.common = MLPLayer(self.hidden_size, self.hidden_size, layer_N=0, use_orthogonal=self._use_orthogonal, activation_id=self._activation_id)
        
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        # value
        if self._use_popart:
            self.v_out = init_(PopArt(input_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(input_size, 1))

        # action
        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(self.device)

    def get_actions(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        x = obs
        x = self.obs_prep(x)
        
        # common
        actor_features = self.common(x)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        x = obs
        x = self.obs_prep(x)

        actor_features = self.common(x)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        
        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, available_actions, active_masks)
       
        return action_log_probs, dist_entropy

    def get_values(self, share_obs, rnn_states, masks):
        share_obs = check(share_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        share_x = share_obs
        share_x = self.share_obs_prep(share_x)

        critic_features = self.common(share_x)   
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        
        values = self.v_out(critic_features)

        return values, rnn_states
