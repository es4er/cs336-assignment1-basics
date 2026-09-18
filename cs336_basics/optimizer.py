import torch 
import math
from torch.optim import Optimizer
from collections.abc import Iterable

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        # 1. 基本参数检查
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        # 2. 将超参数传入 defaults  字典
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """执行单步优化更新"""
        loss = None

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # 3. 状态初始化(第一次运行步时执行)
                if len(state) == 0:
                    state['step'] = 0
                    # m: 一阶矩(梯度的指数移动平均)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # v: 二阶矩(梯度平方的指数移动平均)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                # 4. 更新矩估计(Algorithm 1)
                # m = beta1 * m + (1-beta1) * g
                exp_avg.mul_(beta1).add_(grad, alpha = 1 - beta1)
                # v = beta2 * v + (1-beta2) * g^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2)

                # 5. 计算偏差矫正后的学习率 alpha_t
                # 这一步是为了消除初始值 0 带来的偏移
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = lr * (math.sqrt(bias_correction2) / bias_correction1)

                # 6. 更新参数： theta = theta - alpha_t * m / (sqrt(v) + eps)
                denom = exp_avg_sq.sqrt().add_(eps)
                # 这是一个专门为优化器设计的复合算子，名字可以拆解为：add(加) + constant(常量) + div(除)
                # p.addcdiv_(tensor1, tensor2, value=1.0)  p = p+value*(tensor1 / tensor2)
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # 7. 应用解耦的权重衰减( AdamW的核心特性 )
                # theta = theta - alpha * lambda * theta
                # p.add_(other, alpha=1.0)  p=p+(alpha*other)
                if wd != 0:
                    p.add_(p, alpha=-lr * wd)

        return loss
    