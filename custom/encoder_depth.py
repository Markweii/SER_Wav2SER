import torch
import torch.nn as nn
from torch.nn.functional import leaky_relu
from speechbrain.nnet.normalization import BatchNorm1d

class Encoder(nn.Module):
    def __init__(
        self,
        n_neurons,
        input_shape=None,
        input_size=None,
        activation=nn.LeakyReLU,
        combine_dims=False,
        out_channels=256,
    ):
        super().__init__()
        self.combine_dims = combine_dims

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        # Initialize depthwise separable convolutions
        self.depthwise_separable_conv = nn.Sequential(
            # Depthwise convolution
            nn.Conv1d(input_size, input_size, kernel_size=3, padding=1, groups=input_size),
            nn.BatchNorm1d(num_features=input_size),
            activation(),
            
            # Pointwise convolution to increase channel size to out_channels
            nn.Conv1d(input_size, out_channels, kernel_size=1),
            nn.BatchNorm1d(num_features=out_channels),
            activation(),

            # Another depthwise + pointwise combination if needed
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.BatchNorm1d(num_features=out_channels),
            activation(),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),  # Adjust the output channels as needed
            nn.BatchNorm1d(num_features=out_channels),
            activation(),
        )

    def forward(self, x):
        if x.ndim == 4 and self.combine_dims:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        
        return self.depthwise_separable_conv(x)
