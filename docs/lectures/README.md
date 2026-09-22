# CS336 课程笔记

Stanford CS336《Language Modeling from Scratch》的听课笔记，按讲次整理。这五讲是
Assignment 1 的直接前置知识——Lec 1 交代 Tokenization 的动机，Lec 2 提供算力/显存
估算的语言，Lec 3 给出架构与超参的默认配方，Lec 4 和 Lec 5 讨论 Transformer 在
效率方向上的替代方案与硬件基础。

笔记为中文，含 41 张从讲义截取的配图（存放在 [`assets/`](assets/)）。

| 讲次 | 主题 | 主要内容 |
|---|---|---|
| [Lec 1](lec01-overview-and-tokenization.md) | Overview & Tokenization | 课程动机、当前 LM 版图、课程大纲、Tokenization |
| [Lec 2](lec02-pytorch-einops-and-resource-accounting.md) | Resource Accounting | 显存核算、算力（FLOPs）核算、训练总开销估算 |
| [Lec 3](lec03-architectures-and-hyperparameters.md) | Architectures & Hyperparameters | 从原始 Transformer 到现代 LLM、归一化与残差、FFN 与激活函数、位置编码、超参经验规律、训练稳定性、注意力效率变体 |
| [Lec 4](lec04-attention-alternatives-and-moe.md) | Attention Alternatives & MoE | 替代/稀疏注意力、稀疏专家模型、Routing、MoE 训练挑战、Upcycling、DeepSeek MoE 演进 |
| [Lec 5](lec05-gpus-and-tpus.md) | GPUs & TPUs | GPU 硬件与编程模型、Roofline 分析、常用优化技巧、FlashAttention |

## 说明

- 笔记按课程内容顺序组织，公式与推导保留了原始记录。
- 配图来自课程讲义，版权归 Stanford CS336 所有，仅作个人学习记录使用。
- 与本仓库实现直接对应的推导见 [Assignment 1 学习笔记](../notes/assignment1.md)。
