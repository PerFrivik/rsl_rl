#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import unpad_trajectories


# per frivik

class ActorCriticDoubleConvolutionRNNImage(nn.Module):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        image_input_dims=(1, 40, 64),
        depth_input_dims=(1, 21, 21),
        rnn_type="lstm",
        rnn_hidden_size=256,
        rnn_num_layers=1,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        self.image_input_dims = image_input_dims
        self.depth_input_dims = depth_input_dims


        # CNN for image observations
        self.conv_image_net = nn.Sequential(
            nn.Conv2d(image_input_dims[0], 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((2, 1)),
            nn.Flatten(start_dim=1),
            # output size is 128
        )

        # TODO: Change the sizes here 
        self.conv_depth_net = nn.Sequential(
            nn.Conv2d(depth_input_dims[0], 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((2, 2)),
            nn.Flatten(start_dim=1),
            # output size is 256
        )

        self.cnn_image_output_size = self._compute_conv_image_output_size(image_input_dims) # 128
        self.cnn_depth_output_size = self._compute_conv_depth_output_size(depth_input_dims) # 256

        self.num_image_features = self.image_input_dims[0] * self.image_input_dims[1] * self.image_input_dims[2]
        self.num_depth_features = self.depth_input_dims[0] * self.depth_input_dims[1] * self.depth_input_dims[2]

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs

        # actor gets the image input 
        self.mlp_input_dim_actor_image = (num_actor_obs - self.num_image_features) + self.cnn_image_output_size 
        self.mlp_input_dim_critic_depth = (num_critic_obs - self.num_depth_features) + self.cnn_depth_output_size

        self.memory_image = Memory(self.mlp_input_dim_actor_image, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_depth = Memory(4, type=rnn_type, num_layers=rnn_num_layers, hidden_size=4)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(rnn_hidden_size, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(self.mlp_input_dim_critic_depth, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)


        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Conv Image Network: {self.conv_image_net}")
        print(f"Conv Depth Network: {self.conv_depth_net}")
        print(f"Actor RNN: {self.memory_image}")
        print(f"Critic RNN: {self.memory_depth}")


        # Action noise
        # self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        self.memory_image.reset(dones)  # reset hidden states of the RNN
        self.memory_depth.reset(dones)  # reset hidden states of the RNN

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def update_image_input_dims(self, observations, masks, hidden_states):
        # num_image_features = self.image_input_dims[0] * self.image_input_dims[1] * self.image_input_dims[2]
        # on batch_mode: observation: (L, B, D)
        batch_model = masks is not None
        if batch_model:
            L, B, D = observations.size()
        else:
            print("DEBUG: non-batch mode observations.size()", observations.size())
        # Split the observations tensor
        other_obs = observations[..., :-self.num_image_features]  # First part: Non-image features
        image_obs = observations[..., -self.num_image_features:]  # Last part: Image features

        image_obs = image_obs.reshape(-1, *self.image_input_dims)  # Reshape image observations to 4D
        image_features = self.conv_image_net(image_obs)  # CNN output, flattened to 1D

        if batch_model:
            image_features = image_features.view(L, B, -1)
            
        # Concatenate image features with other (non-image) observations
        print("image_features", image_features.size())
        print("other_obs", other_obs.size())
        print("--------------------")
        
        combined_features = torch.cat((image_features, other_obs), dim=-1)
        
        combined_features = self.memory_image(combined_features, masks, hidden_states) #TODO: change memory input dim to 128 + other_obs.size()

        return combined_features.squeeze(0)  # Remove the time dimension
    
    def update_depth_input_dims(self, observations, masks, hidden_states):
        # num_depth_features = self.image_input_dims[0] * self.image_input_dims[1] * self.image_input_dims[2]
        batch_model = masks is not None
        if batch_model:
            L, B, D = observations.size()
        else:
            print("DEBUG: non-batch mode for Depth observations.size()", observations.size())

        # Split the observations tensor
        other_obs = observations[..., :-self.num_depth_features]  # First part: Non-depth features
        depth_obs = observations[..., -self.num_depth_features:]  # Last part: depth features

        depth_obs = depth_obs.reshape(-1, *self.depth_input_dims)  # Reshape depth observations to 4D
        depth_features = self.conv_depth_net(depth_obs)  # CNN output, flattened to 1D
        
        if batch_model:
            depth_features = depth_features.view(L, B, -1)

        # Concatenate depth features with other (non-depth) observations
        print("depth_features", depth_features.size())
        print("other_obs", other_obs.size())
        combined_features = torch.cat((depth_features, other_obs), dim=-1)
        # fake memory
        fake_feat = self.memory_depth(combined_features[..., :4], masks, hidden_states) #TODO: change memory input dim to 256 + other_obs.size()
        # if not fake_feat.size()[1] == combined_features.size()[1]:
        #     print("\033[91mfake_feat.size()\033[0m", fake_feat.size())
        #     print("\033[91mcombined_features.size()\033[0m", combined_features.size()) 
            
        # unpad_trajectories
        combined_features = unpad_trajectories(combined_features, masks) if batch_model else combined_features
        return combined_features
    
    # def update_observation_space(self, observations, masks, hidden_states):

    #     # state 2 only the actor has the image input 
    #     print("observations.size()[1]", observations.size()[1])
    #     # DEBUG: if observations.size()[1] == 0:
    #     if observations.size()[1] == 0:
    #         print("DEBUG: observations.size()[1] == 0")
    #         print("observations.size()", observations.size())
    #     if observations.size()[1] == self.num_actor_obs:
                
    #         return self.update_image_input_dims(observations, masks, hidden_states)

    #     # state 3 only the critic has the image input
    #     # if observations.size()[1] == self.num_critic_obs:
    #     else:

    #         return self.update_depth_input_dims(observations, masks, hidden_states)


    def update_distribution(self, combined_features):
        # Pass the combined features through the actor network
        mean = self.actor(combined_features)
        print("mean.size()", mean.size())
        self.distribution = Normal(mean, self.log_std.exp().expand_as(mean))


    def act(self, observations, masks=None, hidden_states=None):
        combined_features = self.update_image_input_dims(observations, masks, hidden_states)
        self.update_distribution(combined_features)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    # def act_inference(self, observations):
    #     actions_mean = self.actor(observations)
    #     return actions_mean

    def act_inference(self, observations, masks=None, hidden_states=None):
        combined_features = self.update_image_input_dims(observations, masks, hidden_states)
        # Pass the combined features through the actor network
        actions_mean = self.actor(combined_features)
        return actions_mean


    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        combined_features = self.update_depth_input_dims(critic_observations, masks, hidden_states)
        # Pass the combined features through the critic network
        value = self.critic(combined_features)
        return value
    
    def get_hidden_states(self):
        print("DEBUG: get hidden states")
        return self.memory_image.hidden_states, self.memory_depth.hidden_states
    
    def _compute_conv_image_output_size(self, shape):
        # Compute the CNN output size only once during initialization
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)  # Batch size of 1, with input shape
            output = self.conv_image_net(dummy_input)
            return int(torch.flatten(output, 1).size(1))
        
    def _compute_conv_depth_output_size(self, shape):
        # Compute the CNN output size only once during initialization
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)  # Batch size of 1, with input shape
            output = self.conv_depth_net(dummy_input)
            return int(torch.flatten(output, 1).size(1))


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

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