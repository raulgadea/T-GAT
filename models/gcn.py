import argparse
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """
    Implementation of the GCN for spatio-temporal forecasting

    Args:
        in_channels: Model input length
        out_channels: Model output length
        improved: If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
        **kwargs (optional): Additional arguments of GCN.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = True, **kwargs):
        """

        Args:
            in_channels:
            out_channels:
            improved:
            **kwargs:
        """
        super().__init__()
        self._in_channels = in_channels  # seq_len for prediction
        self._out_channels = out_channels  # hidden_dim for prediction
        self._improved = improved
        self.conv = GCNConv(in_channels, out_channels, improved=improved)

    def forward(self, x, edge_index, edge_weight):
        """
        Forward propagation

        Args:
            x: input matrix
            edge_index: Sparse adjacency matrix
            edge_weight: Sparse weight adjacency matrix

        Returns: Forward output

        """
        # (batch_size*num_nodes, seq_len)
        ax = self.conv(x, edge_index, edge_weight)
        # (batch_size*num_nodes, output_dim)
        outputs = torch.tanh(ax)
        # act(AXW) (num_nodes * batch_size, output_dim)
        return outputs

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        """
        Add specific class arguments for parsing
        Args:
            parent_parser: Previous parser

        Returns: Updated purser

        """
        parent_parser.add_argument("--hid_channels", type=int, default=64)
        parent_parser.add_argument("--improved", type=bool, default=True)
        return parent_parser

    @property
    def hyperparameters(self):
        """
        Returns: Model hyperparameters

        """
        return {
            "in_channels": self._in_channels,
            "out_channels": self._out_channels,
            "improved": self._improved,
        }
