import time
import os
import pickle
import sys
import random


def load_pkl(pkl_dir):
    all_data = []
    pkl_files = os.listdir(pkl_dir)
    T0 = time.perf_counter()
    for file_name in pkl_files:
        batch_file = os.path.join(pkl_dir, file_name)
        with open(batch_file, 'rb') as f:
            data = pickle.load(f)
            all_data.extend(data)
    T1 = time.perf_counter()
    print(f"读取时间 {T1 - T0} s")
    return all_data

def list_to_pkl(data_list:list, output_folder:str, batch_size = 10000):
    os.makedirs(output_folder, exist_ok=True)
    # 分批保存
    total_batches = (len(data_list) + batch_size - 1) // batch_size  # 计算总批次
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data_list))  # 确保索引不超出范围
        batch_data = data_list[start_idx:end_idx]  # 获取当前批次数据

        # 构造文件名并保存
        output_file = os.path.join(output_folder, f"batch_{i + 1}.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(batch_data, f)
        print(f"Saved batch {i + 1} with {len(batch_data)} entries to {output_file}")

def split_list(data, train_ratio = 0.9, seed = None):
    if seed is not None:
        random.seed(seed)

    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    train_size = int(len(shuffled_data) * train_ratio)
    train_data = shuffled_data[:train_size]
    valid_data = shuffled_data[train_size:]

    return train_data, valid_data

def data_divide(all_data, train_ratio = 0.9, seed = 42):
    trainlist, validlist = split_list(all_data, train_ratio=train_ratio, seed=seed)
    list_to_pkl(trainlist, "data/train", 10000)
    list_to_pkl(validlist, "data/eval", 10000)

data_divide(load_pkl("pkls"))