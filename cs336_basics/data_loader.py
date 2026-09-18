import torch
import numpy as np
import numpy.typing as npt

def get_batch(
        dataset: npt.NDArray,
        batch_size: int,
        max_seq_length: int,
        device: str
)-> tuple[torch.Tensor, torch.Tensor]:
    """
    随机采样一个训练批次

    返回：
        x: 输入张量，形状[batch_size, max_seq_length]
        y: 目标张量，形状[batch_size, max_seq_length]
    """
    n = len(dataset)
    # 最后一个可用的起点，必须留出 max_seq_length 给 x ,再多留一位给 y
    max_idx = n - max_seq_length - 1

    # 随机选取 batch_size 个起始点
    ix = torch.randint(0, max_idx + 1, (batch_size,))

    # 提取序列并转为 Numpy 数组，再转为 Tensor
    # 这样做比循环里逐个 to(device) 快得多
    x = torch.stack([torch.from_numpy(dataset[i : i + max_seq_length].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(dataset[i+1 : i + max_seq_length + 1].astype(np.int64)) for i in ix])


    # 一次性搬运到 GPU
    return x.to(device), y.to(device)

