import numpy as np
from math import log2

data = np.array([
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
])


def calc_entropy(data):
    # 计算数据集的熵
    label_col = data[:, -1]
    _, counts = np.unique(label_col, return_counts=True)
    probs = counts / len(label_col)
    entropy = sum(-p * log2(p) for p in probs)
    return entropy


def calc_info_gain(data, col_idx):
    # 计算指定列的信息增益
    col = data[:, col_idx]
    entropy = calc_entropy(data)
    # 计算每个类别的出现次数与概率
    vals, counts = np.unique(col, return_counts=True)
    probs = counts / len(col)
    # 计算条件熵
    cond_entropy = sum(
        p * calc_entropy(data[col == v, :]) for v, p in zip(vals, probs))
    # 计算信息增益
    info_gain = entropy - cond_entropy
    return info_gain


# 计算每个特征的信息增益
num_cols = data.shape[1] - 1
info_gains = [calc_info_gain(data, i) for i in range(num_cols)]
print(info_gains)
