#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.actor_critic_double_convolution import ActorCriticDoubleConvolution, get_activation
from rsl_rl.utils import unpad_trajectories

# per frivik

class ActorCriticDoubleConvolutionRNN(ActorCriticDoubleConvolution):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_size=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        image_input_dims=(1, 40, 64),
        depth_input_dims=(1, 21, 21),
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            image_input_dims=image_input_dims,
            depth_input_dims=depth_input_dims,
            **kwargs,
        )

        activation = get_activation(activation)

        self.memory_a = Memory(self.mlp_input_dim_actor_image, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(self.mlp_input_dim_critic_depth, type=rnn_type, num_layers=rnn_num_layers, hidden_size=8)

        print(f"Image RNN: {self.memory_a}")
        print(f"Depth RNN: {self.memory_c}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def update_image_input_dims(self, observations):
        # num_image_features = self.image_input_dims[0] * self.image_input_dims[1] * self.image_input_dims[2]

        # Split the observations tensor
        other_obs = observations[:, :-self.num_image_features]  # First part: Non-image features
        image_obs = observations[:, -self.num_image_features:]  # Last part: Image features

        image_obs = image_obs.view(-1, *self.image_input_dims)  # Reshape image observations to 4D
        image_features = self.conv_image_net(image_obs)  # CNN output, flattened to 1D

        # Concatenate image features with other (non-image) observations
        # combined_features = torch.cat((image_features, other_obs), dim=-1)
        feat_list = [image_features, other_obs]
        return feat_list
    
    def update_observation_space(self, observations, masks=None, hidden_states=None):

        # state 2 only the actor has the image input 
        if observations.size()[1] == self.num_actor_obs:
            feat_list = self.update_image_input_dims(observations)
            image_feat, other_feat = feat_list[0], feat_list[1]
            image_feat = self.memory_a(image_feat, masks, hidden_states)
            combined_feat = torch.cat((image_feat.squeeze(0), other_feat), dim=1)
            return combined_feat

        # state 3 only the critic has the image input
        # if observations.size()[1] == self.num_critic_obs:
        else:
            feat_list = self.update_image_input_dims(observations)
            depth_feat, other_feat = feat_list[0], feat_list[1]
            depth_feat = self.memory_c(depth_feat, masks, hidden_states)
            combined_feat = torch.cat((depth_feat.squeeze(0), other_feat), dim=1)
            return combined_feat
            # return super().update_depth_input_dims(observations)

    def act(self, observations, masks=None, hidden_states=None):
        combined_features = self.update_observation_space(observations, masks, hidden_states)
        super().update_distribution(combined_features)
        return self.distribution.sample()
    
    def act_inference(self, observations):
        combined_features = self.update_observation_space(observations, None, None)

        # Pass the combined features through the actor network
        actions_mean = self.actor(combined_features)

        return actions_mean
    
    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        combined_features = self.update_observation_space(critic_observations, masks, hidden_states)
        
        # Pass the combined features through the critic network
        value = self.critic(combined_features)
        return value
    
    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states

class Memory(torch.nn.Module):
    def __init__(self, input_size, type="lstm", num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0