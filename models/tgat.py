import torch
from torch_geometric.nn import GATv2Conv


class TGATCell(torch.nn.Module):
    """
    Implementation of the TGAT Cell for spatio-temporal forecasting

    Args:
        in_channels: Model input length
        out_channels: Model output length
        heads: Number of multi-head-attentions.
        concat: If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
        dropout: Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training.
        share_weights: If set to :obj:`True`, the same matrix
            will be applied to the source and the target node of every edge.
        **kwargs (optional): Additional arguments of GATv2.
    """

    def __init__(
            self,
            batch_size: int,
            in_channels: int,
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            dropout: float = 0.,
            share_weights: bool = False,
            **kwargs

    ):
        super().__init__()
        if concat:
            assert out_channels % heads == 0, 'incorrect number heads'
            hid_channels = out_channels // heads
        else:
            hid_channels = out_channels
        self._batch_size = batch_size
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._heads = heads
        self._concat = concat
        self._dropout = dropout
        self._share_weights = share_weights

        self.conv1 = GATv2Conv(in_channels + out_channels, hid_channels * 2, heads=heads, concat=concat,
                               dropout=dropout,
                               share_weights=share_weights, **kwargs)

        self.conv2 = GATv2Conv(in_channels + out_channels, hid_channels, heads=heads, concat=concat, dropout=dropout,
                               share_weights=share_weights, **kwargs)

    def forward(self, x, edge_index, edge_weight, h):
        """
        Forward propagation

        Args:
            x: input matrix
            edge_index: Sparse adjacency matrix
            edge_weight: Sparse weight adjacency matrix
            h: Hidden state of previous cell

        Returns: Forward output

        """
        # Concatenation 1
        cat1 = torch.cat((x, h), dim=1)

        # Convolution 1
        ru = torch.sigmoid(self.conv1(cat1, edge_index))

        # r, u
        ru = ru.reshape((self._batch_size, -1, 2 * self._out_channels)).reshape((self._batch_size, -1))
        r, u = torch.chunk(ru, chunks=2, dim=1)
        r = r.reshape((self._batch_size, -1, self._out_channels)).reshape((-1, self._out_channels))
        u = u.reshape((self._batch_size, -1, self._out_channels)).reshape((-1, self._out_channels))

        # Concatenation 2
        cat2 = torch.cat((x, r * h), dim=1)

        # c
        c = torch.tanh(self.conv2(cat2, edge_index))

        # h
        h = u * h + (1.0 - u) * c
        return h

    @property
    def hyperparameters(self):
        """
        Returns: Model hyperparameters

        """
        return {"in_channels": self._in_channels, "out_channels": self._out_channels}


class TGAT(torch.nn.Module):
    """
    Implementation of the TGAT for spatio-temporal forecasting

    Args:
        in_channels: Model input length
        out_channels: Model output length
        heads: Number of multi-head-attentions.
        concat: If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
        dropout: Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training.
        share_weights: If set to :obj:`True`, the same matrix
            will be applied to the source and the target node of every edge.
        **kwargs (optional): Additional arguments of GATv2.
    """

    def __init__(self,
                 batch_size: int,
                 hid_channels: int,
                 heads: int = 1,
                 concat: bool = True,
                 dropout: float = 0.,
                 share_weights: bool = False,
                 **kwargs
                 ):
        super().__init__()

        self._batch_size = batch_size
        self._hid_channels = hid_channels
        self._heads = heads
        self._concat = concat
        self._dropout = dropout
        self._share_weights = share_weights
        self.gatcell = TGATCell(batch_size, 1, hid_channels, heads=heads, concat=concat, dropout=dropout,
                                share_weights=share_weights, **kwargs)

    def forward(self, x, edge_index, edge_weight):
        """
        Forward propagation

        Args:
            x: input matrix
            edge_index: Sparse adjacency matrix
            edge_weight: Sparse weight adjacency matrix

        Returns: Forward output

        """
        batch_nodes = x.size(0)
        pre_len = x.size(1)
        h = torch.zeros(batch_nodes, self._hid_channels).type_as(x)

        for step in range(pre_len):
            xt = x[:, step].unsqueeze(1)
            h = self.gatcell(xt, edge_index, edge_weight, h)
        return h

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        """
        Add specific class arguments for parsing
        Args:
            parent_parser: Previous parser

        Returns: Updated purser

        """
        parent_parser.add_argument("--hid_channels", type=int, default=64)
        parent_parser.add_argument("--heads", type=int, default=2)
        parent_parser.add_argument("--dropout", type=float, default=0.)
        parent_parser.add_argument("--concat", type=bool, default=True)
        parent_parser.add_argument("--share_weights", type=bool, default=False)
        return parent_parser

    @property
    def hyperparameters(self):
        """
        Returns: Model hyperparameters

        """
        return {
            "batch_size": self._batch_size,
            "hid_channels": self._hid_channels,
            "heads": self._heads,
            "concat": self._concat,
            "dropout": self._dropout,
            "share_weights": self._share_weights
        }
