import pytorch_lightning as pl
import models
import tasks
import datamodules
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy.linalg as la
import math

DATA_PATHS = {
    "shenzhen": {"feat": "data/sz_speed.csv", "adj": "data/sz_adj.csv"},
    "losloop": {"feat": "data/los_speed.csv", "adj": "data/los_adj.csv"},
    "m30": {"feat": "data/m30_speed.csv", "adj": "data/m30_speed_adj.csv"},
    "madrid": {"feat": "data/madrid_intensity.csv", "adj": "data/madrid_adj.csv"},
}


def get_model(args, dm):
    """
    Get Model for training

    Args:
        args: Parsed arguments
        dm: Data module

    Returns: Model

    """
    model = None
    if args.model_name == "GCN":
        model = models.GCN(in_channels=args.seq_len, out_channels=args.hid_channels, improved=args.improved)
    if args.model_name == "GRU":
        model = models.GRU(batch_size=args.batch_size, hid_channels=args.hid_channels)
    if args.model_name == "TGCN":
        model = models.TGCN(adj=dm.adj, hidden_dim=args.hidden_dim)
    if args.model_name == "TGAT":
        model = models.TGAT(batch_size=args.batch_size, hid_channels=args.hid_channels, heads=args.heads,
                            concat=args.concat, dropout=args.dropout, share_weights=args.share_weights)
    if args.model_name == "GAT":
        model = models.GAT(in_channels=args.seq_len, out_channels=args.hid_channels, heads=args.heads,
                           concat=args.concat, dropout=args.dropout, share_weights=args.share_weights)
    return model


def save_nn_results(results, args):
    """
    Save testing metrics in csv

    Args:
        results: Result metrics
        args: Parsed arguments

    """
    f = int(args.pre_len / 3) if args.data == 'losloop' else args.pre_len
    path = f'{args.result_path}/{args.data}/{f}'
    if not os.path.exists(path):
        os.makedirs(path)
    if args.model_name in ['TGAT', 'TGATv2', 'TGAT', 'GAT']:
        filename = f'{path}/{args.model_name}_{args.data}_{args.hid_channels}_{args.dropout}_{args.weight_decay}.csv'
    elif args.model_name in ['GCNo', 'TGCN']:
        filename = f'{path}/{args.model_name}_{args.data}_{args.hidden_dim}.csv'
    else:
        filename = f'{path}/{args.model_name}_{args.data}_{args.hid_channels}_{args.weight_decay}.csv'
    df = pd.DataFrame(results)
    df.to_csv(filename, sep='|', index=False)


def main_tgcn(args):
    """
    Main program for TGCN model

    Args:
        args: Parsed arguments

    Returns: Result metrics

    """
    dm = datamodules.TGCNDataModule(
        feat_path=DATA_PATHS[args.data]["feat"], adj_path=DATA_PATHS[args.data]["adj"], **vars(args)
    )
    model = get_model(args, dm)
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="train_loss")
    task = tasks.TGCNForecastTask(model=model, feat_max_val=dm.feat_max_val, **vars(args))
    # trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(task, dm)
    trainer.test(dataloaders=dm, ckpt_path='best')
    results = task.data
    return results


def main_gat(args):
    """
    Main program for Pytorch geometric models

    Args:
        args: Parsed arguments

    Returns: Result metrics

    """
    dm = datamodules.GATDataModule(
        feat_path=DATA_PATHS[args.data]["feat"], adj_path=DATA_PATHS[args.data]["adj"], **vars(args)
    )
    model = get_model(args, dm)
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="train_loss")
    task = tasks.GATForecastTask(model=model, feat_max_val=dm.feat_max_val, **vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
    # trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    trainer.fit(task, dm)
    trainer.test(dataloaders=dm, ckpt_path='best')
    results = task.data
    return results


def preprocess_data(data, time_len, rate, seq_len, pre_len):
    """
    Create training and test data

    Args:
        data: Feature data
        time_len: Time period length
        rate: train test split rate
        seq_len: Model input length
        pre_len: Model output length

    Returns: Training input, Training output, Test input, Test output

    """
    data1 = np.mat(data)
    train_size = int(time_len * rate)
    train_data = data1[0:train_size]
    test_data = data1[train_size:time_len]

    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0: seq_len])
        trainY.append(a[seq_len: seq_len + pre_len])
    for i in range(len(test_data) - seq_len - pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0: seq_len])
        testY.append(b[seq_len: seq_len + pre_len])
    return trainX, trainY, testX, testY


def evaluation(a, b):
    """
    Evaluation metrics
    Args:
        a: predicted value
        b: true value

    Returns: RMSE, MAE, F,  R2, VAR

    """
    rmse = math.sqrt(mean_squared_error(a, b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a - b) / la.norm(a)
    r2 = 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()
    var = 1 - (np.var(a - b)) / np.var(a)
    return rmse, mae, 1 - F_norm, r2, var


def save_baselines_results(rmse, mae, acc, r2, var, args):
    """
    Save testing metrics in csv

    Args:
        rmse: Root Mean Squared Error
        mae: Mean Absolute Error
        acc: Accuracy
        r2: R2 information
        var: Explained variance
        args: Parsed arguments

    """
    results = {'val_loss': [0],
               'RMSE': [rmse],
               'MAE': [mae],
               'accuracy': [acc],
               'R2': [r2],
               'ExplainedVar': [var],
               }
    f = int(args.pre_len / 3) if args.data == 'losloop' else args.pre_len
    filename = f'results/{args.data}/{f}/{args.method}_{args.data}.csv'
    df = pd.DataFrame(results)
    df.to_csv(filename, sep='|', index=False)