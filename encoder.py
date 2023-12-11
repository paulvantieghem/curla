# This piece of code was copied & modified from the following source:
#
#    Title: CURL: Contrastive Unsupervised Representation Learning for Sample-Efficient Reinforcement Learning
#    Author: Laskin, Michael and Srinivas, Aravind and Abbeel, Pieter
#    Date: 2020
#    Availability: https://github.com/MishaLaskin/curl

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple

class CNNEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape: Tuple, feature_dim: int, pretrained: bool = True):
        super().__init__()

        # Assert correctness of input shape
        assert len(obs_shape) == 3, "'obs_shape' must be 3 dimensional (C, H, W)!"
        assert obs_shape[0] == 3, "'obs_shape' must have 3 channels!"
        assert obs_shape[1]%64 == 0, "Image height must be divisible by 64!"
        assert obs_shape[2]%64 == 0, "Image width must be divisible by 64!"

        # Initialize class variables
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.pretrained = pretrained

        # Use ResNet-18 as the backbone
        resnet18 = models.resnet18(pretrained=self.pretrained)

        # Get only the convolutional layers from ResNet-18
        self.convs = nn.Sequential(*list(resnet18.children())[:-2])

        # Use Global Average Pooling to reduce the number of parameters
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.feature_dim)

    def forward(self, obs, detach: bool = False):

        # Preprocess the observation
        obs = obs / 255.0

        # Pass the observation through the convolutional layers
        features = self.convs(obs)

        # Detach the features if needed
        if detach:
            features = features.detach()

        # Pass the features through the average pooling layer
        x = self.avgpool(features)

        # Flatten the features
        x = torch.flatten(x, 1)

        # Pass the features through the fully connected layer
        latent = self.fc(x)

        return latent

    def copy_conv_weights_from(self, source):
        """Tie all convolutional layers of the encoder network with the source encoder network."""
        assert isinstance(source, CNNEncoder), "Source must be an instance of CNNEncoder!"
        for layer_self, layer_source in zip(self.convs, source.convs):
            if isinstance(layer_self, nn.Conv2d) and isinstance(layer_source, nn.Conv2d):
                # Copy the weights and biases
                layer_self.weight.data.copy_(layer_source.weight.data)
                if layer_self.bias is not None:
                    layer_self.bias.data.copy_(layer_source.bias.data)


