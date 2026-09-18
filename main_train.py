import argparse
import torch
import os
import numpy as np
import wandb
import time
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.clip_gradient_norm import clip_gradient_norm
from cs336_basics.get_lr_cosine_schedule import get_lr_cosine_schedule
from cs336_basics.data_loader import get_batch
from cs336_basics.loss_function import cross_entropy
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint


def estimate_loss(model, train_data, val_data, batch_size, context_length, device, eval_iters=20):
    """
    在多个 batch 上计算平均训练损失和验证损失， 减少单个 batch 带来的随机波动
    """
    model.eval()

    losses = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        loss_list = []

        with torch.no_grad():
            for _ in range(eval_iters):
                x, y = get_batch(data, batch_size, context_length, device)
                logits = model(x)
                loss = cross_entropy(logits, y)
                loss_list.append(loss.item())
        losses[split] = sum(loss_list)/len(loss_list)

    model.train()
    return losses["train"], losses["val"]


def main():
    parser = argparse.ArgumentParser()
    # --- 模型基础超参数 ---
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=256) # 相当于 max_seq_length
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=10000)

    # --- 消融实验开关 ---
    # Ablation 1: 移除 RMSNorm
    parser.add_argument("--no_rms_norm", action="store_true", help="Disable RMSNorm")
    # Ablation 2: Pre-norm vs Post-norm
    parser.add_argument("--norm_mode", type=str, default="pre", choices=["pre","post"], help="Using pre_norm, disable post_norm")
    # Ablation 3: 移除 RoPE (NoPE)
    parser.add_argument("--no_rope", action="store_true", help="Disable Rotary Positional Embedding")
    # Ablation 4: SwiGLU vs SiLU
    parser.add_argument("--ffn_type", type=str, default="swiglu", choices=["swiglu", "silu"], help="Using swiglu, disable silu")


    # --- 优化器超参数 ---
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--warmup_iters",type=int, default=1000)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--max_norm", type=float, default=1.0)

    # --- Evaluation超参数 ---
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_iters", type=int, default=20)


    # --- 路径与系统 ---
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--valid_data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")


    # --- WandB 设置 ---
    parser.add_argument("--run_name", type=str, default=None, help="WandB 实验名称")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. 加载数据 (使用 memmap )
    # 假设数据是以 uint16 存储的二进制文件
    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"Training data not found at {args.train_data_path}")
    if not os.path.exists(args.valid_data_path):
        raise FileNotFoundError(f"Validation data not found at {args.valid_data_path}")

    # np.memmap 延迟加载数据到内存，非常适合大数据集，并且将二进制文件转为 dtype (uint16) 数组
    train_data = np.memmap(args.train_data_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(args.valid_data_path, dtype=np.uint16, mode='r')

    print(f"训练集大小：{len(train_data)} tokens")
    print(f"验证集大小：{len(val_data)} tokens")

    # 2. 处理消融实验逻辑
    # 如果 no_rope 为 True, 则 theta 设为 None, TransformerBlock 内部就不会初始化 RoPE
    actual_rope_theta = None if args.no_rope else 10000.0
    # use_rms_norm 逻辑取反
    use_rms_norm = not args.no_rms_norm

    # 3. 初始化模型
    model = TransformerLM(
        vocab_size=args.vocab_size,
        max_seq_len=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=actual_rope_theta,
        device=args.device,
        # 传入实验参数
        use_rms_norm=use_rms_norm,
        norm_mode=args.norm_mode,
        ffn_type=args.ffn_type
    ).to(args.device)

    print(f"Model Config: Norm={args.norm_mode}, useNorm={use_rms_norm}, FFN={args.ffn_type}")

    # 4. 初始化优化器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    # 5. 检查点恢复逻辑
    start_iters = 0
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    if os.path.exists(ckpt_path):
        start_iters = load_checkpoint(ckpt_path, model, optimizer)
        print(f"Resuming from iteration {start_iters}")

    # 6. 初始化 WandB 监控
    wandb.init(
        project="cs336-assignment1",
        name=args.run_name,
        config=args,
    )

    # 记录训练开始时间
    train_start_time = time.time()

    # 主训练循环
    for it in range(start_iters, args.max_iters):
        # A. 更新学习率
        lr = get_lr_cosine_schedule(it,args.lr, args.min_lr, args.warmup_iters, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # B. 训练步
        model.train()
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)

        logits = model(x)
        loss = cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        clip_gradient_norm(model.parameters(), args.max_norm)

        optimizer.step()


        elapsed_time = time.time() - train_start_time
        # C. 验证与日志记录
        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            train_eval_loss, val_loss = estimate_loss(model, train_data, val_data, args.batch_size, args.context_length, args.device, args.eval_iters)
            print(
                f"Iter {it}"
                f"train_loss {train_eval_loss:.4f},"
                f"val_loss {val_loss:.4f},"
                f"time {elapsed_time:.2f}s"
            )

            wandb.log({
                "train/loss": train_eval_loss,
                "val/loss": val_loss,
                "lr": lr,
                "iters": it+1,
                "elapsed_time":elapsed_time
            })

            
        # D. 保存检查点 (每 1000 步保存一次)
        if it % 1000 == 0 and it > 0:
            save_checkpoint(model, optimizer, it, ckpt_path)

    # 训练结束保存最终模型
    save_checkpoint(model, optimizer, args.max_iters, os.path.join(args.out_dir, "ckpt_final"))
    wandb.finish()

if __name__ == "__main__":
    main()










