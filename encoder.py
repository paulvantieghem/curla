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
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.resnet import BasicBlock

class CNNEncoder(nn.Module):
    def __init__(self, obs_shape: Tuple, latent_dim: int):
        super(CNNEncoder, self).__init__()
        self.obs_shape = obs_shape
        self.latent_dim = latent_dim

        # Assert correctness of input shape
        assert len(obs_shape) == 3, "'obs_shape' must be 3 dimensional (C, H, W)!"
        assert obs_shape[0]%3 == 0, "Number of channels must be divisible by 3!"

        # Get a pretrained ResNet-18 model
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Modify the first convolutional layer to take in 9-channel images
        model.conv1 = nn.Conv2d(self.obs_shape[0], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Convolutional layers
        self.convs = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )

        # Latent head
        self.latent_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(512, latent_dim),
        )

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
            # Convulutional layers can be nested in Sequential layers within BasicBlock layers
            if isinstance(layer_self, nn.Conv2d) and isinstance(layer_source, nn.Conv2d):
                # Copy the weights and biases
                layer_self.weight.data.copy_(layer_source.weight.data)
                if layer_self.bias is not None:
                    layer_self.bias.data.copy_(layer_source.bias.data)

            # Should take BasicBlock layers into account
            elif isinstance(layer_self, nn.Sequential) and isinstance(layer_source, nn.Sequential):
                if isinstance(layer_self[0], BasicBlock) and isinstance(layer_source[0], BasicBlock):
                    # Copy conv1 and conv2
                    layer_self[0].conv1.weight.data.copy_(layer_source[0].conv1.weight.data)
                    layer_self[0].conv2.weight.data.copy_(layer_source[0].conv2.weight.data)
                    
# Get torch device
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

if __name__ == "__main__":

    # Define the encoder
    frame_stack = 3
    n_channels = 3 * frame_stack
    height = 90
    width = 160
    obs_shape = (n_channels, height, width)
    latent_dim = 50
    model = CNNEncoder(obs_shape=obs_shape, latent_dim=latent_dim)
    device = get_device()
    model.to(device)

    # Print the model summary
    batch_size = 256
    input_size = (batch_size, n_channels, height, width)
    try:
        import torchinfo
        dummy_input = torch.randn(input_size).to(device)
        output = model(dummy_input)
        print(output.shape)
        torchinfo.summary(
            model, 
            input_size=input_size,
            device=device,
        )
    except:
        print("torchinfo is not installed. Skipping summary.")

    # Test copy_conv_weights_from feature
    model2 = CNNEncoder(obs_shape=obs_shape, latent_dim=latent_dim)
    model2.to(device)
    model.copy_conv_weights_from(model2)
    print("\nWeights of the convolutional layers of both models are equal: ")
    for layer_self, layer_source in zip(model.convs, model2.convs):
        if isinstance(layer_self, nn.Conv2d) and isinstance(layer_source, nn.Conv2d):
            print("1:", torch.equal(layer_self.weight.data, layer_source.weight.data))
        elif isinstance(layer_self, nn.Sequential) and isinstance(layer_source, nn.Sequential):
            print(layer_self._get_name())
            if isinstance(layer_self[0], BasicBlock) and isinstance(layer_source[0], BasicBlock):
                print("2:", torch.equal(layer_self[0].conv1.weight.data, layer_source[0].conv1.weight.data))
                print("3:", torch.equal(layer_self[0].conv2.weight.data, layer_source[0].conv2.weight.data))

