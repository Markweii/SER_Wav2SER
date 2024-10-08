import torch
import torch.nn as nn
import speechbrain as sb
from speechbrain.nnet.pooling import StatisticsPooling
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.normalization import LayerNorm


class Encoder(nn.Module):
    """Computes a linear transformation y = wx + b.

    Arguments
    ---------
    n_neurons : int
        It is the number of output neurons (i.e, the dimensionality of the
        output).
    input_shape: tuple
        It is the shape of the input tensor.
    input_size: int
        Size of the input tensor.
    bias : bool
        If True, the additive bias b is adopted.
    combine_dims : bool
        If True and the input is 4D, combine 3rd and 4th dimensions of input.

    Example
    -------
    >>> inputs = torch.rand(10, 50, 40)
    >>> lin_t = Linear(input_shape=(10, 50, 40), n_neurons=100)
    >>> output = lin_t(inputs)
    >>> output.shape
    torch.Size([10, 50, 100])
    """

    def __init__(
        self,
        n_neurons,
        input_shape=None,
        input_size=None,
        activation=torch.nn.ReLU,
        combine_dims=False,
        out_channels = 256,
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
        """Returns the linear transformation of input tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input to transform linearly.
        """
        if x.ndim == 4 and self.combine_dims:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        return self.group_conv(x)