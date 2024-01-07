# Small parts of this code were copied & modified from the following source:
#
#    Title: CURL: Contrastive Unsupervised Representation Learning for Sample-Efficient Reinforcement Learning
#    Author: Laskin, Michael and Srinivas, Aravind and Abbeel, Pieter
#    Date: 2020
#    Availability: https://github.com/MishaLaskin/curl

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Basic block for ResNet
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# CNN encoder based on the encoder of a ResNet-18 model, but modified
# to take in 9-channel images as input
class CNNEncoder(nn.Module):
    def __init__(self, obs_shape: Tuple, latent_dim: int):
        super(CNNEncoder, self).__init__()
        self.obs_shape = obs_shape
        self.latent_dim = latent_dim

        # Assert correctness of input shape
        assert len(obs_shape) == 3, "'obs_shape' must be 3 dimensional (C, H, W)!"
        assert obs_shape[0]%3 == 0, "Number of channels must be divisible by 3!"
        assert obs_shape[1]%64 == 0, "Image height must be divisible by 64!"
        assert obs_shape[2]%64 == 0, "Image width must be divisible by 64!"

        # Initial channels of the ResNet-18 model
        self.in_channels = 64

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(obs_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # Average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, latent_dim)

        # Convolutional layers
        self.convs = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        )
        
        # Latent head
        self.latent_head = nn.Sequential(
            self.avgpool,
            nn.Flatten(start_dim=1, end_dim=-1),
            self.fc,
        )

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, detach: bool = False):

        # Preprocess the observation
        obs = obs / 255.0

        # Pass the observation through the convolutional layers
        features = self.convs(obs)

        # Detach the gradients of the features if specified
        if detach:
            features = features.detach()

        # Pass the features through the latent head
        latent = self.latent_head(features)

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


if __name__ == "__main__":

    # Define the encoder
    frame_stack = 3
    n_channels = 3 * frame_stack
    height = 128
    width = 256
    obs_shape = (n_channels, height, width)
    latent_dim = 64
    model = CNNEncoder(obs_shape=obs_shape, latent_dim=latent_dim)

    # Print the model summary
    batch_size = 256
    input_size = (batch_size, n_channels, height, width)
    try:
        import torchinfo
        torchinfo.summary(model, input_size=input_size)
    except:
        print("Could not print model summary!")

    # Test copy_conv_weights_from feature
    model2 = CNNEncoder(obs_shape=obs_shape, latent_dim=latent_dim)
    model.copy_conv_weights_from(model2)
    print("\nWeights of the convolutional layers of both models are equal: ", end="")
    for layer_self, layer_source in zip(model.convs, model2.convs):
        if isinstance(layer_self, nn.Conv2d) and isinstance(layer_source, nn.Conv2d):
            print(torch.equal(layer_self.weight.data, layer_source.weight.data))