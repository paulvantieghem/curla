# This piece of code was copied & modified from the following source:
#
#    Title: CURL: Contrastive Unsupervised Representation Learning for Sample-Efficient Reinforcement Learning
#    Author: Laskin, Michael and Srinivas, Aravind and Abbeel, Pieter
#    Date: 2020
#    Availability: https://github.com/MishaLaskin/curl

import torch
import torch.nn as nn


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias

# Calculate output dimensions for different input sizes: http://layer-calc.com/, 
# or just use the commented print statement on line 86 of this file

# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}

# for 76 x 135 inputs
OUT_DIM_RECT_76_135 = {4: [31, 61],}

# for 90 x 160 inputs
OUT_DIM_RECT_90_160 = {4: [38, 73],}


class CNNEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32, output_logits=False):
        super().__init__()
        assert len(obs_shape) == 3

        if obs_shape[1:] == (84, 84):
            out_dim = OUT_DIM[num_layers]
        elif obs_shape[1:] == (64, 64):
            out_dim = OUT_DIM_64[num_layers]
        elif obs_shape[1:] == (76, 135) and num_layers == 4:
            out_dim = OUT_DIM_RECT_76_135[num_layers]
        elif obs_shape[1:] == (90, 160) and num_layers == 4:
            out_dim = OUT_DIM_RECT_90_160[num_layers]
        else:
            raise NotImplementedError("Encoder does not support input shape")

        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        # Build convolutional layers
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=obs_shape[0], 
                                              out_channels=num_filters, 
                                              kernel_size=3, 
                                              stride=2)])

        for _ in range(num_layers - 1):
            self.convs.append(nn.Conv2d(in_channels=num_filters, 
                                        out_channels=num_filters, 
                                        kernel_size=3, 
                                        stride=1))

        # Build linear layers
        self.fc = nn.Linear(num_filters * out_dim[0] * out_dim[1], self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_logits = output_logits

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            # print(f'conv{i+1} shape: {conv.shape}')
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)

if __name__ == "__main__":

    # Get torch device
    def get_device():
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        return device

    # Define the encoder
    frame_stack = 3
    n_channels = 3 * frame_stack
    height = 90
    width = 160
    obs_shape = (n_channels, height, width)
    feature_dim = 50
    model = CNNEncoder(obs_shape=obs_shape, feature_dim=feature_dim, num_layers=4, num_filters=32, output_logits=False)
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