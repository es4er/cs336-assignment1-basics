import math

def get_lr_cosine_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int
)->float:
    """
    计算第 it 次迭代时, 带预热的余弦退火学习率

    参数：
        it: 当前迭代步数 (t)
        max_learning_rate: 学习率的峰值 (alpha_max)
        min_learning_rate: 学习率的底值 (alpha_min)
        warmup_iters: 预热阶段总步数 (T_w)
        cosine_cycle_iters: 整个衰减周期结束的步数 (T_c)
    """

    # 1. 预热阶段：线性增长逻辑
    if it < warmup_iters:
        # 从 0 匀速增长到 max_ learning_rate
        return max_learning_rate * it / warmup_iters

    # 2. 衰减周期后：维持最小值
    if it > cosine_cycle_iters:
        return min_learning_rate

    # 3. 余弦退火核心逻辑
    # a. 计算当前处于退火阶段的进度百分比 (0.0 到 1.0)
    # it - warmup_iters: 距离预热结束走了多少步
    # cosine_cycle_iters - warmup_iters: 整个退火阶段的总长度
    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)

    # b. 计算余弦系数
    # math.cos(math.pi * decay_ratio):
    # 当进度为 0 时，结果为 cos(0)=1
    # 当进度为 1 时，结果为 cos(pi)=-1
    # coeff = 0.5 * (1 + [-1,1]) -> 范围[0.0, 1.0]
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    # c. 最终计算
    # 学习率从 max 降向 min
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)
