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
        groups=32  # Add a groups parameter
    ):
        super().__init__()
        self.combine_dims = combine_dims

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        # Initialize group convolutions
        self.group_conv = nn.Sequential(
            # Group convolution
            nn.Conv1d(input_size, out_channels, kernel_size=3, padding=1, groups=groups),
            nn.BatchNorm1d(num_features=out_channels),
            activation(),
            
            # Additional group convolution layers
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, groups=groups),
            nn.BatchNorm1d(num_features=out_channels),
            activation(),

            # Adjust the output channels as needed
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(num_features=out_channels),
            activation(),
        )

    def forward(self, x):
        if x.ndim == 4 and self.combine_dims:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        
        return self.group_conv(x)
