import torch
import os 
import typing

def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: typing.Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]]
):
    """
    保存当前训练状态
    """

    # 1. 构建一个包含当前所有必要信息的字典
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }

    # 2. 使用 torch.save 将字典写入目标 (可以是路径或文件夹)
    torch.save(checkpoint, out)


def load_checkpoint(
        src: typing.Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer
) -> int:
    """
    从检查点的状态恢复，并返回保存时的迭代步数
    """

    # 1. 加载字典
    # 使用 map_location = 'cpu', 可以防止在没有 GPU 的机器上加载时报错
    checkpoint = torch.load(src, map_location='cpu')

    # 2. 恢复模型权重
    model.load_state_dict(checkpoint['model_state_dict'])

    # 3. 恢复优化器状态(动量、步数等)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 4. 返回保存时的迭代步数
    return checkpoint['iteration']


