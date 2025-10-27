import time
import math
import torch
import numpy as np
from utils.utils import log_string, metric, load_data


def test(args, log,device):
    #加载数据
    (trainX, trainY, trainTE, valX, valY, valTE, testX, testY, testTE,
     query_trainX, query_trainY, query_valX, query_valY, query_testX, query_testY,
     periods, strengths, adj_matrix, SE, mean, std,
     train_aux_his, train_sim, val_aux_his, val_sim, test_aux_his, test_sim,
     train_aux_pred, val_aux_pred, test_aux_pred) = load_data(args, device)

    num_train, _, num_vertex = trainX.shape
    num_val = valX.shape[0]
    num_test = testX.shape[0]
    train_num_batch = math.ceil(num_train / args.batch_size)
    val_num_batch = math.ceil(num_val / args.batch_size)
    test_num_batch = math.ceil(num_test / args.batch_size)

    #加载模型
    log_string(log, '**** testing model ****')
    log_string(log, f'loading model from {args.model_file}')
    model = torch.load(args.model_file)
    model.eval()
    log_string(log, 'model loaded!')
    log_string(log, 'evaluating...')

    #测试函数
    def evaluate(dataX, dataTE, dataY, query_data, aux_data_his,aux_data_pred, sim_data, num_batches, dataset_name):
        predictions = []
        actuals = []

        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * args.batch_size
                end_idx = min(dataX.shape[0], (batch_idx + 1) * args.batch_size)

                X = dataX[start_idx:end_idx].to(device)
                TE = dataTE[start_idx:end_idx].to(device)
                label = dataY[start_idx:end_idx].to(device)
                query = query_data[start_idx:end_idx].to(device)
                aux_his = aux_data_his[start_idx:end_idx].to(device)
                aux_pred = aux_data_pred[start_idx:end_idx].to(device)
                sim = sim_data[start_idx:end_idx].to(device)

                pred = model(X, TE, query, aux_his,aux_pred, sim)  # 传入辅助信息
                pred = pred * std + mean

                predictions.append(pred.detach().cpu())
                actuals.append(label.detach().cpu())

                del X, TE, label, query, pred

        predictions = torch.cat(predictions, dim=0)
        actuals = torch.cat(actuals, dim=0)

        MAE_steps, RMSE_steps, MAPE_steps = [], [], []

        for step in range(args.num_pred):
            pred_step = predictions[:, step]
            actual_step = actuals[:, step]
            mae_step, rmse_step, mape_step = metric(pred_step, actual_step)
    
            MAE_steps.append(mae_step)
            RMSE_steps.append(rmse_step)
            MAPE_steps.append(mape_step)

        avg_mae = np.mean(MAE_steps)
        avg_rmse = np.mean(RMSE_steps)
        avg_mape = np.mean(MAPE_steps)

        log_string(log, f'{dataset_name} MAE: {avg_mae:.4f}, RMSE: {avg_rmse:.4f}, MAPE: {avg_mape * 100:.2f}%')
        log_string(log, f'performance in each prediction step ({dataset_name})')
        for step in range(args.num_pred):
            log_string(log, f'step {step + 1}: MAE={MAE_steps[step]:.4f}, RMSE={RMSE_steps[step]:.4f}, MAPE={MAPE_steps[step] * 100:.2f}%')

        log_string(log, f'average: MAE={avg_mae:.4f}, RMSE={avg_rmse:.4f}, MAPE={avg_mape * 100:.2f}%')

        return predictions

    #训练集评估
    start_test = time.time()
    # 修改调用部分
    trainPred = evaluate(trainX, trainTE, trainY, query_trainX, train_aux_his, train_aux_pred,train_sim, train_num_batch, 'train')
    valPred = evaluate(valX, valTE, valY, query_valX, val_aux_his, val_aux_pred, val_sim, val_num_batch, 'val')
    testPred = evaluate(testX, testTE, testY, query_testX, test_aux_his,test_aux_pred, test_sim, test_num_batch, 'test')
    end_test = time.time()

    log_string(log, f'total testing time: {end_test - start_test:.1f}s')

    return trainPred, valPred, testPred