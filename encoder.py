# This piece of code was copied/modified from the following source:
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

# Calculate output dimensions for different input sizes: http://layer-calc.com/

# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}

# for 76 x 135 inputs
OUT_DIM_RECT_76_135 = {4: [31, 61],}

# for 90 x 160 inputs
OUT_DIM_RECT_90_160 = {4: [36, 71],}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32,output_logits=False):
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


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters,*args):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, output_logits=False
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, output_logits
    )
