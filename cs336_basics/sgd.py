import torch
import torch.nn as nn
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        """
        params: 传入模型优化所需要的参数(通常是 model.parameters())
        defaults: 一个字典, 存储默认超参数(如学习率 lr)
        调用 super.__init__后, Pytorch 会将参数组织在 self.params_group 中
        """
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self):
        loss = None
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                # 获取该参数对应的状态字典(用于记录步数t)
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0

                t = state["t"]
                grad = p.grad.data

                # 执行更新公式 
                p.data -= lr / math.sqrt(t + 1) * grad

                # 更新步数
                state["t"] += 1

        return loss

