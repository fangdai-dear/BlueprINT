import torch
import torch.nn as nn
import torch.nn.functional as F
from featup.layers import ImplicitFeaturizer, ChannelNorm, MinMaxScaler

class FeatUpModule(nn.Module):
    def __init__(self, in_channels=1024, out_channels=1024, color_feats=True, n_freqs=10):
        super().__init__()
        self.scaler = MinMaxScaler()
        self.featurizer = ImplicitFeaturizer(color_feats=color_feats, n_freqs=n_freqs, learn_bias=True)
        self.norm1 = ChannelNorm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.relu = nn.ReLU()
        self.norm2 = ChannelNorm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.scaler(x)
        x = self.featurizer(x)
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm2(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        return x