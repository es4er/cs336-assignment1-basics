import torch
from collections.abc import Iterable

def clip_gradient_norm(parameters: Iterable[torch.nn.Parameter], max_norm: float):
    """
    实现全局梯度裁剪 (Global Norm Clipping)

    参数:
        parameters: 模型的所有参数 (model.parameters())
        max_norm: 允许的最大梯度 L2 范数 (M)
    """

    # 1. 过滤掉没有梯度的参数 (防止对 None 对象操作)
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return

    # 2. 计算全局 L2 范数 (Global L2 Norm)
    total_norm = 0.0
    for p in params_with_grad:
        # 使用 .detach() 极其重要
        # 梯度裁剪是在计算完导数之后进行的数值操作，我们不希望"计算范数"的过程也被计入计算图
        # torch.norm(..., p=2) 算出当前梯度的 L2 范数 L_i
        param_norm = torch.norm(p.grad.detach(), p=2)

        # 将各层的范数平方累加 
        total_norm += param_norm.item() ** 2

    total_norm = total_norm ** 0.5


    # 3. 检查是否触发裁剪
    eps = 1e-6  # 防止除零的稳定性参数
    if total_norm > max_norm:
        # 计算统一的缩放系数
        clip_coef = max_norm / (total_norm + eps)

        # 4. 原地修改每个参数的梯度
        # 使用 mul_ 直接修改内存，不产生临时副本，节省显存
        for p in params_with_grad:
            p.grad.detach().mul_(clip_coef)