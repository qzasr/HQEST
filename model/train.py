import time
import datetime
import torch
import torch.optim as optim
import math
from utils.utils import log_string, metric, load_data
from model.model import HQEST


def train(model, args, log, loss_criterion, optimizer, scheduler,device):
    (trainX, trainY, trainTE, valX, valY, valTE, testX, testY, testTE,
     query_trainX, query_trainY, query_valX, query_valY, query_testX, query_testY,
     periods, strengths, adj_matrix, SE, mean, std,
     train_aux_his, train_sim, val_aux_his, val_sim, test_aux_his, test_sim,
     train_aux_pred, val_aux_pred, test_aux_pred) = load_data(args, device)

    num_train, _, num_vertex = trainX.shape
    log_string(log, '**** training model ****')
    num_val = valX.shape[0]
    train_num_batch = math.ceil(num_train / args.batch_size)
    val_num_batch = math.ceil(num_val / args.batch_size)

    wait = 0
    val_loss_min = float('inf')
    best_model_wts = None
    train_total_loss = []
    val_total_loss = []

    for epoch in range(args.max_epoch):
        if wait >= args.patience:
            log_string(log, f'early stop at epoch: {epoch:04d}')
            break

        permutation = torch.randperm(num_train)
        trainX = trainX[permutation]
        trainTE = trainTE[permutation]
        trainY = trainY[permutation]
        query_trainX = query_trainX[permutation]
        train_aux_his=train_aux_his[permutation]
        train_aux_pred=train_aux_pred[permutation]
        train_sim=train_sim[permutation]

        start_train = time.time()
        model.train()
        train_loss = 0

        for batch_idx in range(train_num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_train, (batch_idx + 1) * args.batch_size)

            X = trainX[start_idx:end_idx].to(device)
            TE = trainTE[start_idx:end_idx].to(device)
            label = trainY[start_idx:end_idx].to(device)
            query = query_trainX[start_idx:end_idx].to(device)
            aux_his = train_aux_his[start_idx:end_idx].to(device)
            aux_pred = train_aux_pred[start_idx:end_idx].to(device)
            sim = train_sim[start_idx:end_idx].to(device)

            optimizer.zero_grad()
            pred = model(X, TE, query, aux_his,aux_pred, sim) 

            pred = pred * std + mean
            loss_batch = loss_criterion(pred, label)

            loss_batch.backward()
            optimizer.step()

            train_loss += float(loss_batch) * (end_idx - start_idx)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            del X, TE, label, query, pred, loss_batch

        train_loss /= num_train
        train_total_loss.append(train_loss)
        end_train = time.time()

        start_val = time.time()
        val_loss = 0
        model.eval()

        with torch.no_grad():
            for batch_idx in range(val_num_batch):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
                X = valX[start_idx:end_idx].to(device)
                TE = valTE[start_idx:end_idx].to(device)
                label = valY[start_idx:end_idx].to(device)
                query = query_valX[start_idx:end_idx].to(device)
                aux_his = val_aux_his[start_idx:end_idx].to(device)
                aux_pred = val_aux_pred[start_idx:end_idx].to(device)
                sim = val_sim[start_idx:end_idx].to(device)
                pred = model(X, TE, query, aux_his, aux_pred, sim)

                pred = pred * std + mean
                loss_batch = loss_criterion(pred, label)

                val_loss += loss_batch.item() * (end_idx - start_idx)
                del X, TE, label, query, pred, loss_batch

        val_loss /= num_val
        val_total_loss.append(val_loss)
        end_val = time.time()

        log_string(
            log,
            '%s | epoch: %04d/%d, training time: %.1fs, inference time: %.1fs' %
            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1,
             args.max_epoch, end_train - start_train, end_val - start_val))
        log_string(
            log, f'train loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')

        if val_loss <= val_loss_min:
            log_string(
                log,
                f'val loss decrease from {val_loss_min:.4f} to {val_loss:.4f}, saving model to {args.model_file}')
            wait = 0
            val_loss_min = val_loss
            best_model_wts = model.state_dict()
        else:
            wait += 1

        scheduler.step()

    model.load_state_dict(best_model_wts)
    torch.save(model, args.model_file)
    log_string(log, f'Training completed. Best model saved to {args.model_file}')

    return train_total_loss, val_total_loss