import torch 


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    计算数值稳定的交叉熵损失。

    参数：
        logits: 形状为(Batch, Seq, Vocab_size) 的预测分量
        targets: 形状为(Batch, Seq) 的真实 Token ID 
    """ 

    # 1. 计算每组 logits 的最大值 M , 用于数值稳定
    # dim=-1 表示在词表维度搜索， keepdim = True 保证结果形状为 (Batch, Seq, 1)
    # 这样在后续执行 'logits - m' 时可触发自动广播
    m = torch.max(logits, dim=-1, keepdim=True).values

    # 2. 提取目标位置对应的原始分值 o_y
    # 使用 gather 函数从词表维度中根据 targets 提取对应的分值
    # 由于 gather 要求 index 的维度与输入一致， 需要将 targets 升维成 (Batch, Seq, 1)
    target_logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # 3. 计算 Log-Sum-Exp
    shifted_logits = logits - m
    # 注意 m 提取后形状是 (B, S, 1), 求和形状为 (B, S), 相加时需先 squeeze m
    log_sum_exp = m.squeeze(-1) + torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))

    # 4. 计算每个 Token 的独立损失值
    loss = log_sum_exp - target_logits

    # 5. 按照作业要求，对整个批次求平均值，返回一个标量
    return torch.mean(loss)

