import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_info
import models
import tasks
import datamodules
import pandas as pd
import os
from utils import get_model, main_gat, main_tgcn, save_nn_results


def main(args):
    rank_zero_info(vars(args))
    results = globals()["main_" + args.settings](args)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--data",
                        type=str,
                        help="The name of the dataset",
                        choices=("shenzhen", "losloop", "m30", "m302"),
                        default="m30")
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model for spatiotemporal prediction",
        choices=("GCN", "GRU", "TGCN", "TGAT", "GAT"),
        default="GAT",
    )
    parser.add_argument(
        "--settings",
        type=str,
        help="The type of settings, e.g. tgc learning",
        choices=("gat", "tgcn"),
        default="gat",
    )
    parser.add_argument("--log_path", type=str, default='lightning_logs', help="Path to the output console log file")
    parser.add_argument("--log_name", type=str, default=None, help="Name of the log directory")
    parser.add_argument("--result_path", type=str, default="results", help="Path to results")

    temp_args, _ = parser.parse_known_args()
    if temp_args.log_name is None:
        log_name = temp_args.data
    else:
        log_name = temp_args.log_name
    logger = TensorBoardLogger(temp_args.log_path, name=log_name, default_hp_metric=False)

    parser = getattr(datamodules, temp_args.settings.upper() + "DataModule").add_data_specific_arguments(parser)
    parser = getattr(models, temp_args.model_name).add_model_specific_arguments(parser)
    parser = getattr(tasks, temp_args.settings.upper() + "ForecastTask").add_task_specific_arguments(parser)

    args = parser.parse_args()
    args.logger = logger
    args.gpus = [args.gpus]
    if args.log_name is None:
        args.log_name = f'logs_{args.data}'
    results = main(args)

    if args.result_path:
        save_nn_results(results, args)
