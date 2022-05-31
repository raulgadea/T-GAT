# Attention to Traffic Forecasting: Improving Predictions with Temporal Graph Attention Networks #

This is the PyTorch implementation of the paper: [Attention to Traffic Forecasting: Improving Predictions with Temporal Graph Attention Networks](https://www.techrxiv.org/articles/preprint/Attention_to_Traffic_Forecasting_Improving_Predictions_with_Temporal_Graph_Attention_Networks/19732483)

Use baselines.py for HA and SVR models and main.py for Neural Netowrk models.

## Requirements ##
- numpy
- pandas
- pytorch_lightning
- scikit_learn
- torch
- torch_geometric
- torchmetrics

## Example ##

`python main.py --model_name TGAT --max_epochs 3000 --learning_rate 0.001 --weight_decay 0 --gradient_clip_val 0 --batch_size 64 --hid_channels 128 --heads 2 --dropout 0.1 --loss mse_with_regularizer --seq_len 4 --data m30 --split_ratio 0.8 --result_path results  --settings gat --log_path logs  --gpus 2 --pre_len `
