#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

# per frivik

class ActorCriticDoubleConvolution(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        image_input_dims=(1, 36, 64),
        depth_input_dims=(1, 21, 21),
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
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(start_dim=1),
        )

        # TODO: Change the sizes here 
        self.conv_depth_net = nn.Sequential(
            nn.Conv2d(depth_input_dims[0], 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(start_dim=1),
        )

        self.cnn_image_output_size = self._compute_conv_image_output_size(image_input_dims)
        self.cnn_depth_output_size = self._compute_conv_depth_output_size(depth_input_dims)

        self.num_image_features = self.image_input_dims[0] * self.image_input_dims[1] * self.image_input_dims[2]
        self.num_depth_features = self.depth_input_dims[0] * self.depth_input_dims[1] * self.depth_input_dims[2]

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs

        # actor gets the image input 
        self.mlp_input_dim_actor_image = (num_actor_obs - self.num_image_features) + self.cnn_image_output_size
        self.mlp_input_dim_critic_depth = (num_critic_obs - self.num_depth_features) + self.cnn_depth_output_size

        # # Here I'm guessing that the image is in both the actor and critic observations
        # if num_actor_obs - num_critic_obs < 0.1:
        #     self.mlp_input_dim_actor_image = (num_actor_obs - num_image_features) + self.cnn_depth_output_size
        #     self.mlp_input_dim_critic_depth = (num_critic_obs - num_image_features) + self.cnn_depth_output_size
        #     self.state = 1
        # # Here I'm guessing that the image is only in the actor observations
        # elif num_actor_obs > num_critic_obs:
        #     self.mlp_input_dim_actor_image = (num_actor_obs - num_image_features) + self.cnn_depth_output_size
        #     self.mlp_input_dim_critic_depth = num_critic_obs
        #     self.state = 2
        # # Here I'm guessing that the image is only in the critic observations
        # else:
        #     self.mlp_input_dim_actor_image = num_actor_obs
        #     self.mlp_input_dim_critic_depth = (num_critic_obs - num_image_features) + self.cnn_depth_output_size
        #     self.state = 3


        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(self.mlp_input_dim_actor_image, actor_hidden_dims[0]))
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
        pass

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
    
    def update_image_input_dims(self, observations):
        # num_image_features = self.image_input_dims[0] * self.image_input_dims[1] * self.image_input_dims[2]

        # Split the observations tensor
        other_obs = observations[:, :-self.num_image_features]  # First part: Non-image features
        image_obs = observations[:, -self.num_image_features:]  # Last part: Image features

        image_obs = image_obs.view(-1, *self.image_input_dims)  # Reshape image observations to 4D
        image_features = self.conv_image_net(image_obs)  # CNN output, flattened to 1D

        # Concatenate image features with other (non-image) observations
        combined_features = torch.cat((image_features, other_obs), dim=-1)
        return combined_features
    
    def update_depth_input_dims(self, observations):
        # num_depth_features = self.image_input_dims[0] * self.image_input_dims[1] * self.image_input_dims[2]

        # Split the observations tensor
        other_obs = observations[:, :-self.num_depth_features]  # First part: Non-depth features
        depth_obs = observations[:, -self.num_depth_features:]  # Last part: depth features

        depth_obs = depth_obs.view(-1, *self.depth_input_dims)  # Reshape depth observations to 4D
        depth_features = self.conv_depth_net(depth_obs)  # CNN output, flattened to 1D

        # Concatenate depth features with other (non-depth) observations
        combined_features = torch.cat((depth_features, other_obs), dim=-1)
        return combined_features
    
    def update_observation_space(self, observations):

        # state 2 only the actor has the image input 
        if observations.size()[1] == self.num_actor_obs:
                
            return self.update_image_input_dims(observations)

        # state 3 only the critic has the image input
        # if observations.size()[1] == self.num_critic_obs:
        else:

            return self.update_depth_input_dims(observations)


    def update_distribution(self, observations):
        # Process image observations with CNN
        # ogga 
        # Calculate the total number of image features (flattened)

        combined_features = self.update_observation_space(observations)

        # Pass the combined features through the actor network
        mean = self.actor(combined_features)
        self.distribution = Normal(mean, mean * 0.0 + self.log_std.exp())


    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    # def act_inference(self, observations):
    #     actions_mean = self.actor(observations)
    #     return actions_mean

    def act_inference(self, observations):

        combined_features = self.update_observation_space(observations)

        # Pass the combined features through the actor network
        actions_mean = self.actor(combined_features)

        return actions_mean


    def evaluate(self, observations, **kwargs):

        # num_image_features = self.image_input_dims[0] * self.image_input_dims[1] * self.image_input_dims[2]

        # # Split the observations tensor
        # other_obs = observations[:, :-num_image_features]  # First part: Non-image features
        # image_obs = observations[:, -num_image_features:]  # Last part: Image features

        # image_obs = image_obs.view(-1, *self.image_input_dims)

        # # Process image observations with CNN
        # image_features = self.conv_image_net(image_obs)  # CNN output, flattened to 1D
        # image_features = torch.flatten(image_features, 1)

        # # Concatenate image features with other (non-image) observations
        # combined_features = torch.cat((image_features, other_obs), dim=-1)
        combined_features = self.update_observation_space(observations)

        # Pass the combined features through the critic network
        value = self.critic(combined_features)
        return value
    
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
