import argparse

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import torch

from .spatiotemporal_dt import SpatioTemporalDT


def load_features(feat_path, dtype=np.float32):
    """
    Loads features dataset

    Args:
        feat_path: Path to features file
        dtype: Feature data type

    Returns: Features array

    """

    feat_df = pd.read_csv(feat_path)
    feat = np.array(feat_df, dtype=dtype)
    return feat


def load_adjacency_matrix(adj_path, dtype=np.float32):
    """
    Loads adjacency matrix dataset
     Args:
         adj_path: Path to adjacency matrix file
         dtype: Adjacency matrix data type

     Returns: Adjacency matrix array

     """

    adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df, dtype=dtype)
    return adj


def generate_dataset(
        data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    """
    Generates train and test data

    Args:
        data: Feature data
        seq_len: Model input length
        pre_len: Model output length
        time_len: Time period length
        split_ratio: Train test split radio
        normalize: Normalize the data or not

    Returns: Training input, Training output, Test input, Test output

    """
    if time_len is None:
        time_len = data.shape[0]
    if normalize:
        max_val = np.max(data)
        data = data / max_val
    train_size = int(time_len * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:time_len]
    train_X, train_Y, test_X, test_Y = list(), list(), list(), list()
    for i in range(len(train_data) - seq_len - pre_len):
        train_X.append(np.array(train_data[i: i + seq_len]))
        train_Y.append(np.array(train_data[i + seq_len: i + seq_len + pre_len]))
    for i in range(len(test_data) - seq_len - pre_len):
        test_X.append(np.array(test_data[i: i + seq_len]))
        test_Y.append(np.array(test_data[i + seq_len: i + seq_len + pre_len]))
    return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)


def generate_spatiotemporal_datasets(
        data, adj, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    """
    Generates train and test spatio-temporal datasets

    Args:
        data: Features data
        adj: Adjacency matrix data
        seq_len: Model input length
        pre_len: Model output length
        time_len: Time period length
        split_ratio: Train test split radio
        normalize: Normalize the data or not

    Returns: Train dataset, Test dataset

    """
    train_X, train_Y,test_X, test_Y = generate_dataset(
        data,
        seq_len,
        pre_len,
        time_len=time_len,
        split_ratio=split_ratio,
        normalize=normalize,
    )
    train_dataset = SpatioTemporalDT(torch.FloatTensor(train_X), torch.FloatTensor(train_Y),
                                     torch.FloatTensor(adj))
    test_dataset = SpatioTemporalDT(torch.FloatTensor(test_X), torch.FloatTensor(test_Y), torch.FloatTensor(adj))
    return train_dataset, test_dataset


class GATDataModule(pl.LightningDataModule):
    """
    Pytorch Lighning Datamodule for Pytorch geometric models

    Args:
        feat_path: Path to features file
        adj_path: Path to adjacency matrix file
        batch_size: Batch size
        seq_len: Model input length
        pre_len: Model output length
        split_ratio: Train test split radio
        normalize: Normalize the data or not
        **kwargs: Keyword arguments
    """
    def __init__(self,
                 feat_path: str,
                 adj_path: str,
                 batch_size: int = 64,
                 seq_len: int = 12,
                 pre_len: int = 3,
                 split_ratio: float = 0.8,
                 normalize: bool = True,
                 **kwargs):
        super().__init__()
        self._feat_path = feat_path
        self._adj_path = adj_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.split_ratio = split_ratio
        self.normalize = normalize
        self._feat = load_features(self._feat_path)
        self._feat_max_val = np.max(self._feat)
        self._adj = load_adjacency_matrix(self._adj_path)

    def setup(self, stage: str = None):
        """
        Set up train and validation spatio-temporal datasets

        Args:
            stage: Datahook variable

        Returns:

        """
        (
            self.train_dataset,
            self.val_dataset,
        ) = generate_spatiotemporal_datasets(self._feat, self._adj, self.seq_len, self.pre_len,
                                             split_ratio=self.split_ratio, normalize=self.normalize)

    def train_dataloader(self):
        """
        Returns: Torch Geometric train Dataloader

        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=True)

    def val_dataloader(self):
        """
        Returns: Torch Geometric validation Dataloader

        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, drop_last=True)

    def test_dataloader(self):
        """
        Returns: Torch Geometric test Dataloader

        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, drop_last=True, shuffle=True)

    @property
    def feat_max_val(self):
        """
        Returns: Maximun feature value for normalization

        """
        return self._feat_max_val

    @property
    def adj(self):
        """
        Returns: Adjacency matrix

        """
        return self._adj

    @staticmethod
    def add_data_specific_arguments(parent_parser):
        """
        Add specific class arguments for parsing
        Args:
            parent_parser: Previous parser

        Returns: Updated purser

        """
        parent_parser.add_argument("--batch_size", type=int, default=32)
        parent_parser.add_argument("--seq_len", type=int, default=12)
        parent_parser.add_argument("--pre_len", type=int, default=3)
        parent_parser.add_argument("--split_ratio", type=float, default=0.8)
        parent_parser.add_argument("--normalize", type=bool, default=True)
        return parent_parser


