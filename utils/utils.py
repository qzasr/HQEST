import pandas as pd
import random
from scipy import signal
import numpy as np
import torch
from scipy.signal import correlate
from scipy.spatial.distance import euclidean
import time
import torch.nn.functional as F

def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

def metric(pred, label):
    mask = torch.ne(label, 0)
    mask = mask.type(torch.float32)
    mask /= torch.mean(mask)
    mae = torch.abs(torch.sub(pred, label)).type(torch.float32)
    rmse = mae ** 2
    mape = mae / label
    mae = torch.mean(mae)
    rmse = rmse * mask
    rmse = torch.sqrt(torch.mean(rmse))
    mape = mape * mask
    mape = torch.mean(mape)
    return mae, rmse, mape


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def masked_mae_loss(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def seq2instance(data, num_his, num_pred):
    num_step, dims = data.shape
    num_sample = num_step - num_his - num_pred + 1
    x = torch.zeros(num_sample, num_his, dims)
    y = torch.zeros(num_sample, num_pred, dims)
    for i in range(num_sample):
        x[i] = data[i: i + num_his]
        y[i] = data[i + num_his: i + num_his + num_pred]
    return x, y

def query2instance(data, num_his, num_pred):
    num_step, row,col,dims = data.shape
    num_sample = num_step - num_his - num_pred + 1
    x = torch.zeros(num_sample, num_his, row,col,dims)
    y = torch.zeros(num_sample, num_pred, row,col,dims)
    for i in range(num_sample):
        x[i] = data[i: i + num_his]
        y[i] = data[i + num_his: i + num_his + num_pred]
    return x, y

def extract_spatiotemporal_periods(data, k):
    
    data = np.nan_to_num(data) 
    data -= np.mean(data, axis=0) 
    total_timesteps = data.shape[0]
    global_magnitude = np.zeros(total_timesteps // 2)
    for node_idx in range(data.shape[1]):
        node_fft = np.fft.fft(data[:, node_idx])
        magnitudes = np.abs(node_fft[:total_timesteps // 2]) * 2 / total_timesteps
        global_magnitude += magnitudes
    global_magnitude /= np.max(global_magnitude)
    peaks, _ = signal.find_peaks(global_magnitude[1:], height=0.1)
    peak_indices = peaks + 1
    top_k_indices = peak_indices[np.argsort(-global_magnitude[peak_indices])[:k]]
    periods = total_timesteps / top_k_indices
    strengths = global_magnitude[top_k_indices]
    sorted_indices = np.argsort(-strengths)
    return periods[sorted_indices], strengths[sorted_indices]

def load_adjacency_matrix(file_path,traffic_data):
    adj_list = pd.read_csv(file_path, header=None, names=['node1', 'node2', 'weight'],sep=r'\s+')
    num_nodes = traffic_data.shape[1]
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for _, row in adj_list.iterrows():
        i, j, w = int(row['node1']), int(row['node2']), float(row['weight'])
        adj_matrix[i, j] = w
        adj_matrix[j, i] = w
    return adj_matrix

def Time_Embedding(num_timesteps, periods):
    TE = np.zeros((num_timesteps, len(periods)), dtype=np.int32)
    for col_idx, period in enumerate(periods):
        if period <= 0:
            raise ValueError(f"Period must be a positive integer {period}")
        TE[:, col_idx] = [t % period for t in range(num_timesteps)]
    return torch.from_numpy(TE)

def calculate_nonlinear_correlation(seq1, seq2):
    seq1 = (seq1 - np.mean(seq1)) / (np.std(seq1) + 1e-8)
    seq2 = (seq2 - np.mean(seq2)) / (np.std(seq2) + 1e-8)
    corr = correlate(seq1, seq2, mode='full')
    return corr

def calculate_region_importance(target_coords, traffic_speed, query_data,window_size=8, step_size=1, max_delay=8,nav_threshold=0.5):
    target_x, target_y = target_coords
    total_time_steps = traffic_speed.shape[0]
    if isinstance(traffic_speed, np.ndarray):
        traffic_speed_arr = traffic_speed
    else:
        traffic_speed_arr = traffic_speed.numpy() if hasattr(traffic_speed, 'numpy') else np.array(traffic_speed)
    if isinstance(query_data, np.ndarray):
        query_arr = query_data
    else:
        query_arr = query_data.numpy() if hasattr(query_data, 'numpy') else np.array(query_data)
    
    target_speed_avg = np.mean(traffic_speed_arr, axis=1)
    
    width, height = query_arr.shape[0], query_arr.shape[1]
    regions = []
    for x in range(width):
        for y in range(height):
            if (x, y) == (target_x, target_y):
                continue
            
            total_nav = np.sum(query_arr[x, y, :, 0]) + np.sum(query_arr[x, y, :, 1])
            
            if total_nav < nav_threshold * total_time_steps:
                continue
            regions.append((x, y))

    
    num_windows = max(1, (total_time_steps - window_size) // step_size + 1)
    region_results = {coord: {'correlations': [], 'importances': []} for coord in regions}
    for w in range(num_windows):
        start_idx = w * step_size
        end_idx = start_idx + window_size
        
        if end_idx > total_time_steps:
            continue
        
        window_speed = target_speed_avg[start_idx:end_idx]
        for coord in regions:
            x, y = coord
            corr = 0
            importance = 0
            for i in range(max_delay + 1):
                if start_idx - i > 0:
                    
                    departures = query_arr[x, y, start_idx-i:end_idx-i, 0]
                    arrivals = query_arr[x, y, start_idx-i:end_idx-i, 1]
                    nav_data = departures + arrivals
                    
                    if np.all(nav_data == 0) or np.var(nav_data) < 1e-6:
                        continue
                    corr1 = calculate_nonlinear_correlation(departures, window_speed)
                    corr2 = calculate_nonlinear_correlation(arrivals, window_speed)
                    
                    signed_max_corr1 = corr1[np.argmax(np.abs(corr1))]
                    signed_max_corr2 = corr2[np.argmax(np.abs(corr2))]
                    
                    corr = (signed_max_corr1 + signed_max_corr2) / 2.0
                    nav_avg = np.mean(nav_data)
                    importance = importance + abs(corr) * nav_avg * (-1 if corr < 0 else 1)
                else:
                    continue
            region_results[coord]['correlations'].append(corr)
            region_results[coord]['importances'].append(importance)

    final_results = []
    for coord, data in region_results.items():
        if not data['importances']:
            continue
        
        avg_corr = np.mean(data['correlations'])
        avg_imp = np.mean(data['importances'])
        coverage = len(data['importances']) / num_windows
        final_results.append((coord, avg_imp, avg_corr, coverage))

    if not final_results:
        print("警告：未找到任何有效区域")
        return []
    final_results.sort(key=lambda x: abs(x[1]), reverse=True)
    top_regions = final_results[:25]
    return top_regions


def process_query_data(query_data):
    is_tensor = torch.is_tensor(query_data)
    num_regions_row, num_regions_col, num_timesteps, _ = query_data.shape
    num_new_timesteps = num_timesteps // 3
    if is_tensor:
        processed_data = torch.zeros((num_new_timesteps, num_regions_row, num_regions_col, 2))
        for t in range(num_new_timesteps):
            start_idx = t * 3
            end_idx = start_idx + 3
            processed_data[t] = query_data[:, :, start_idx:end_idx, :].sum(dim=2)
    else:
        processed_data = np.zeros((num_new_timesteps, num_regions_row, num_regions_col, 2))
        for t in range(num_new_timesteps):
            start_idx = t * 3
            end_idx = start_idx + 3
            processed_data[t] = np.sum(query_data[:, :, start_idx:end_idx, :], axis=2)
    return processed_data


def find_similar_historical_data(data, num_his, num_pred):
    device = data.device
    num_steps, num_vertex = data.shape
    num_samples = num_steps - num_his - num_pred + 1

    similar_history = torch.zeros((num_samples, num_his, num_vertex), device=device)
    future_data = torch.zeros((num_samples, num_pred, num_vertex), device=device)
    similarity_scores = torch.zeros(num_samples, device=device)

    windows = []
    for i in range(num_steps - num_his - num_pred + 1):
        windows.append(data[i:i + num_his])

    if windows:
        windows_flat = torch.stack([w.flatten() for w in windows])
        windows_flat = F.normalize(windows_flat, dim=1)
    else:
        return similar_history, future_data, similarity_scores

    for i in range(num_samples):
        current_window = data[i:i + num_his].flatten()
        current_norm = F.normalize(current_window.unsqueeze(0), dim=1)
        start_idx = max(0, i - 5840)  
        end_idx = max(0, i - 8)  
        num_candidates = end_idx - start_idx
        if num_candidates <= 0:
            continue

        candidate_windows = windows_flat[start_idx:end_idx]
        batch_similarities = torch.mm(current_norm, candidate_windows.T).squeeze(0)

        max_sim, max_idx = torch.max(batch_similarities, dim=0)
        global_idx = start_idx + max_idx.item()

        similarity_scores[i] = max_sim
        similar_history[i] = windows[global_idx]
        future_data[i] = data[global_idx + num_his: global_idx + num_his + num_pred]

    return similar_history, future_data, similarity_scores

def normalize_nonzero(data, mean, std):
    mask = data != 0
    normalized_data = torch.where(mask, (data - mean) / std, data)
    return normalized_data

def load_data(args,device):
    traffic_speed = pd.read_csv(args.traffic_file)
    traffic_speed = torch.tensor(traffic_speed.values)
    periods, strengths = extract_spatiotemporal_periods(traffic_speed, k=3)

    with open(args.SE_file, mode='r') as f:
        lines = f.readlines()
        temp = lines[0].split(' ')
        num_vertex, dims = int(temp[0]), int(temp[1])
        SE = torch.zeros((num_vertex, dims), dtype=torch.float32)
        for line in lines[1:]:
            temp = line.split(' ')
            index = int(temp[0])
            SE[index] = torch.tensor([float(ch) for ch in temp[1:]])

    query_data = np.load(args.query_file)['Query_ST_data']
    query_data = process_query_data(query_data)
    query_data =query_data.transpose((1, 2, 0, 3))
    top_regions = calculate_region_importance((args.row,args.col), traffic_speed, query_data)
    top_coords = [coord for coord, *_ in top_regions[:25]]
    top_data = np.zeros((5, 5, query_data.shape[2], 2), dtype=query_data.dtype)
    region_map = {}
    for idx, (x, y) in enumerate(top_coords):
        row, col = idx // 5, idx % 5
        top_data[row, col] = query_data[x, y]
        region_map[(row, col)] = (x, y)
    query_data = torch.from_numpy(top_data)
    query_data = query_data.permute(2,0,1,3)

    adj_matrix = load_adjacency_matrix(args.adj_file, traffic_speed)

    TE = Time_Embedding(traffic_speed.shape[0], periods)

    num_step = traffic_speed.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps

    train_traffic =traffic_speed[:train_steps]
    val_traffic = traffic_speed[train_steps:train_steps+val_steps]
    test_traffic = traffic_speed[train_steps+val_steps:]

    query_train=query_data[:train_steps]
    query_val = query_data[train_steps:train_steps+val_steps]
    query_test = query_data[train_steps+val_steps:]
    query_trainX, query_trainY = query2instance(query_train, args.num_his, args.num_pred)
    query_valX, query_valY = query2instance(query_val, args.num_his, args.num_pred)
    query_testX, query_testY = query2instance(query_test, args.num_his, args.num_pred)

    query_mean, query_std = torch.mean(query_trainX), torch.std(query_trainX)
    query_trainX = (query_trainX - query_mean) / query_std
    query_valX = (query_valX - query_mean) / query_std
    query_testX = (query_testX - query_mean) / query_std

    trainX, trainY = seq2instance(train_traffic, args.num_his, args.num_pred)
    valX, valY = seq2instance(val_traffic, args.num_his, args.num_pred)
    testX, testY = seq2instance(test_traffic, args.num_his, args.num_pred)

    mean, std = torch.mean(trainX), torch.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std

    train = TE[: train_steps]
    val = TE[train_steps: train_steps + val_steps]
    test = TE[-test_steps:]
    trainTE = seq2instance(train, args.num_his, args.num_pred)
    trainTE = torch.cat(trainTE, 1).type(torch.int32)
    valTE = seq2instance(val, args.num_his, args.num_pred)
    valTE = torch.cat(valTE, 1).type(torch.int32)
    testTE = seq2instance(test, args.num_his, args.num_pred)
    testTE = torch.cat(testTE, 1).type(torch.int32)

    traffic_speed = (traffic_speed - mean)/std
    auxiliary_his_info, auxiliary_pred_info, sim_scores = find_similar_historical_data(
        traffic_speed, args.num_his, args.num_pred
    )
    train_aux_his = auxiliary_his_info[:trainX.shape[0]]
    val_aux_his = auxiliary_his_info[trainX.shape[0]:trainX.shape[0] + valX.shape[0]]
    test_aux_his = auxiliary_his_info[trainX.shape[0] + valX.shape[0]:]

    train_aux_pred = auxiliary_pred_info[:trainX.shape[0]]
    val_aux_pred = auxiliary_pred_info[trainX.shape[0]:trainX.shape[0] + valX.shape[0]]
    test_aux_pred = auxiliary_pred_info[trainX.shape[0] + valX.shape[0]:]

    train_sim = sim_scores[:trainX.shape[0]]
    val_sim = sim_scores[trainX.shape[0]:trainX.shape[0] + valX.shape[0]]
    test_sim = sim_scores[trainX.shape[0] + valX.shape[0]:]

    return [torch.tensor(x).to(device) if isinstance(x, np.ndarray) else x.to(device)
            for x in (trainX, trainY, trainTE, valX, valY, valTE, testX, testY, testTE,
                      query_trainX, query_trainY, query_valX, query_valY, query_testX, query_testY,
                      periods, strengths, adj_matrix, SE, mean, std, train_aux_his, train_sim,
                      val_aux_his, val_sim, test_aux_his, test_sim,train_aux_pred,val_aux_pred,test_aux_pred)]