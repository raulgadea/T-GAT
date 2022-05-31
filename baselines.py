import argparse
import pandas as pd
import numpy as np
from sklearn.svm import NuSVR
from utils import save_baselines_results, evaluation, preprocess_data

DATA_PATHS = {
    "shenzhen": {"feat": "data/sz_speed.csv", "adj": "data/sz_adj.csv"},
    "losloop": {"feat": "data/los_speed.csv", "adj": "data/los_adj.csv"},
    "m30": {"feat": "data/m30_speed.csv", "adj": "data/m30_speed_adj.csv"},
    "madrid": {"feat": "data/madrid_intensity.csv", "adj": "data/madrid_adj.csv"},
}




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        help="The name of the model for spatiotemporal prediction",
        choices=("HA", "SVR"),
        default="SVR",
    )
    parser.add_argument(
        "--data",
        type=str,
        choices=("losloop", "shenzhen", "madrid", "m30"),
        default="los_speed",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--pre_len",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--train_rate",
        type=float,
        default=0.8,
    )

    args = parser.parse_args()
    data_path = DATA_PATHS[args.data]['feat']
    data = pd.read_csv(data_path)

    time_len = data.shape[0]
    num_nodes = data.shape[1]
    train_rate = args.train_rate
    seq_len = args.seq_len
    pre_len = args.pre_len
    trainX, trainY, testX, testY = preprocess_data(data, time_len, train_rate, seq_len, pre_len)
    method = args.method  ####HA or SVR or ARIMA

    ########### HA #############
    if method == 'HA':
        result = []
        for i in range(len(testX)):
            a = np.array(testX[i])
            tempResult = []
            a1 = np.mean(a, axis=0)
            for j in range(pre_len):
                tempResult.append(a1)
                a = a[1:]
                a = np.append(a, [a1], axis=0)
                a1 = np.mean(a, axis=0)
            result.append(tempResult)
        result1 = np.array(result)
        result1 = np.reshape(result1, [-1, num_nodes])
        testY1 = np.array(testY)
        testY1 = np.reshape(testY1, [-1, num_nodes])
        rmse, mae, accuracy, r2, var = evaluation(testY1, result1)
        print('HA_rmse:%r' % rmse,
              'HA_mae:%r' % mae,
              'HA_acc:%r' % accuracy,
              'HA_r2:%r' % r2,
              'HA_var:%r' % var)
        save_baselines_results(rmse, mae, accuracy, r2, var, args)

    ############ SVR #############
    if method == 'SVR':
        total_rmse, total_mae, total_acc, result = [], [], [], []
        for i in range(num_nodes):
            data1 = np.mat(data)
            a = data1[:, i]
            a_X, a_Y, t_X, t_Y = preprocess_data(a, time_len, train_rate, seq_len, pre_len)
            a_X = np.array(a_X)
            a_X = np.reshape(a_X, [-1, seq_len])
            a_Y = np.array(a_Y)
            a_Y = np.reshape(a_Y, [-1, pre_len])
            a_Y = np.mean(a_Y, axis=1)
            t_X = np.array(t_X)
            t_X = np.reshape(t_X, [-1, seq_len])
            t_Y = np.array(t_Y)
            t_Y = np.reshape(t_Y, [-1, pre_len])
            svr_model = NuSVR(kernel='linear')
            # svr_model = SVR(kernel='linear', cache_size=20000)
            svr_model.fit(a_X, a_Y)
            pre = svr_model.predict(t_X)
            pre = np.array(np.transpose(np.mat(pre)))
            pre = pre.repeat(pre_len, axis=1)
            result.append(pre)
            print(i)
        result1 = np.array(result)
        result1 = np.reshape(result1, [num_nodes, -1])
        result1 = np.transpose(result1)
        testY1 = np.array(testY)

        testY1 = np.reshape(testY1, [-1, num_nodes])
        total = np.mat(total_acc)
        total[total < 0] = 0
        rmse1, mae1, acc1, r2, var = evaluation(testY1, result1)
        print('SVR_rmse:%r' % rmse1,
              'SVR_mae:%r' % mae1,
              'SVR_acc:%r' % acc1,
              'SVR_r2:%r' % r2,
              'SVR_var:%r' % var)

        save_baselines_results(rmse1, mae1, acc1, r2, var, args)
