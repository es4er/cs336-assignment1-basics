import torch
import torch.nn as nn
from einops import rearrange
import math
import os 
import typing

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device = None, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        factory_kwargs = {'device': device,'dtype':dtype}

        self.weight = nn.Parameter(torch.empty(in_features,out_features, **factory_kwargs))

        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight,0.0,std=std,a=-3*std,b=3*std)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('...i, io -> ...o', x, self.weight)


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device = None, dtype = None):
        super().__init__()
        # 1. 分配内存并包装为参数(W 维度：vocab_size * d_model)
        factory_kwargs = {'device':device ,'dtype':dtype}
        self.weight = nn.Parameter(torch.empty((num_embeddings,embedding_dim),**factory_kwargs))

        # 2. 按照作业要求进行初始化
        # mean = 0, std = 1.0, 截断在 [-3,3]
        nn.init.trunc_normal_(self.weight,mean=0.0,std=1.0,a=-3.0,b=3.0)

    def forward(self,token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids形状：[B,S]
        # 直接通过索引从矩阵中“捞出”对应的向量
        return self.weight[token_ids] # 返回形状：[B,S,D]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device':device, 'dtype':dtype}

        # 1. 必须初始化为全1 (ones)
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))
        self.eps = eps

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        # x : (batch_size, sequence_length, d_model)

        in_dtype = x.dtype

        # 2. 转换为 float32 以防平方计算时溢出
        x_float = x.to(torch.float32)

        # 3. 计算均方差(Root Mean Square)
        # 公式： rms = sqrt( mean(x^2) + eps)
        # dim = -1 表示在隐藏层维度计算，keepdim = True 方便后续自动广播

        ms = x_float.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(ms + self.eps)

        # 4. 归一化并乘以可学习的增益参数g
        result = (x_float / rms) * self.weight

        return result.to(in_dtype)


def silu_fn(in_features):
    return in_features * torch.sigmoid(in_features)

class SwiGLU(nn.Module):
    def __init__(self, d_model:int, d_ff:int, device=None, dtype=None):
        super().__init__()

        self.d_ff = d_ff
        self.d_model = d_model
        # W1 和 W3 是并行升维层：d_model->d_ff
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        # W2 是降维层：d_ff->d_model
        self.w2 = Linear(d_ff, d_model, device, dtype)


    def forward(self, x:torch.Tensor) -> torch.Tensor:

        gate = silu_fn(self.w1(x))
        signal = self.w3(x)

        return self.w2(gate * signal)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int,device=None):
        super().__init__()
        self.d_k = d_k
        """
        初始化 RoPE 模块
        theta: 基准频率(通常为10000)
        d_k:每个Head的维度(必须是偶数)
        max_seq_len: 最大序列长度
        """

        # 1. 计算频率 omega_k = theta^(-2k / d)
        # 我们只需要计算 d_k/2 个频率，因为旋转是成对进行的
        # arange(0, d_k, 2) 产生 [0, 2, 4, ... , d_k - 2], 对应公式中的 2k-2 (k从1开始)
        powers = torch.arange(0, d_k, 2, device=device).float() / d_k
        freqs = 1.0 / (theta ** powers) # 形状: (d_k/2,)

        # 2. 创建位置序列 [0, 1, 2, ..., max_seq_len - 1]
        t = torch.arange(max_seq_len, device=device).float() # 形状: (max_seq_length,)

        # 3. 计算所有位置的角度 (外积)
        # freqs_matrix 形状：(max_seq_len, d_k/2)
        freqs_matrix = torch.outer(t, freqs)

        # 4. 预计算 cos 和 sin 并作为 buffer 注册
        # 使用 persistent = False 确保这些缓存不会被保存在 state_dict 中(因为可以随时重新生成)
        self.register_buffer("cos_cached", freqs_matrix.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs_matrix.sin(), persistent=False)

    def forward(self,x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 1. 提取 cos/sin (...,Seq, d_k/2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # 2. 维度对齐
        # 只有当 x 是 4D (含 Head 维) 且 cos 是 3D (含 Batch 维)时，才需要手动插入 Head 维
        # 对于 test_rope 这种 3D x vs 2D cos 的情况 ，PyTorch 会自动左侧补1，无需操作
        if x.ndim > cos.ndim and cos.ndim >= 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        # 3. 拆分并旋转
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        output = torch.empty_like(x)
        output[..., 0::2] = x_even * cos - x_odd * sin
        output[..., 1::2] = x_even * sin + x_odd * cos

        return output

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x,dim=dim,keepdim=True)[0]
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    sum_exp = torch.sum(exp_x,dim=dim, keepdim=True)
    result = exp_x / sum_exp

    return result



def scaled_dot_product_attention(Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask:torch.Tensor = None)->torch.Tensor:
    d_k = Q.size(-1)

    scores = torch.einsum('...s d, ... t d -> ... s t',Q,K) / math.sqrt(d_k)

    # 应用因果掩码
    if mask is not None:
        # 将 False 对应的位置设置为负无穷，使其在softmax后的概率为0
        scores = scores.masked_fill(mask == False , float('-inf'))

    # dim = -1 对应的是每一个 Query 对其所有 Key 的分布
    probs = softmax(scores, dim = -1)

    out_put = torch.einsum('... s t, ... t d -> ... s d',probs,V)

    return out_put

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, max_seq_len = None, theta = None, device = None, dtype = None):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model,d_model,device=device,dtype=dtype)
        self.k_proj = Linear(d_model,d_model,device=device,dtype=dtype)
        self.v_proj = Linear(d_model,d_model,device=device,dtype=dtype)

        self.output_proj = Linear(d_model,d_model,device=device,dtype=dtype)

        # 仅在提供 theta 时启用RoPE
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)
        else:
            self.rope = None

    def forward(self,x:torch.Tensor, token_positions:torch.Tensor = None)->torch.Tensor: 
        b,s,d = x.shape

        q = rearrange(self.q_proj(x),'... s (h d) -> ... h s d', h = self.num_heads)
        k = rearrange(self.k_proj(x),'... s (h d) -> ... h s d', h = self.num_heads)
        v = rearrange(self.v_proj(x),'... s (h d) -> ... h s d', h = self.num_heads)

        # 应用 RoPE 
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(s, device=x.device).expand(b,s)

            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # 生成因果掩码
        mask = torch.tril(torch.ones(s,s,device=x.device, dtype=torch.bool))

        # 核心注意力计算
        # 结果形状(Batch, Heads, Seq, d_k)
        attn_out = scaled_dot_product_attention(q,k,v,mask=mask)

        # 合并多头
        attn_out = rearrange(attn_out,'... h s d -> ... s (h d)')

        # 输出投影
        return self.output_proj(attn_out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int,num_heads: int, d_ff: int, max_seq_len: int, theta: float, device=None, dtype=None):
        super().__init__()
        # 初始化因果自注意力模块
        self.attn = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype
        )

        # 初始化两个 RMSNorm 层，分别服务于 Attention 和 FFN
        self.ln1 = RMSNorm(d_model,device=device,dtype=dtype)
        self.ln2 = RMSNorm(d_model,device=device,dtype=dtype)

        # 初始化前馈网络(SwiGLU)
        self.ffn = SwiGLU(d_model,d_ff,device=device,dtype=dtype)

    def forward(self,x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        # 步骤 1 : Attention 子层 (Pre-norm 结构)
        # x 被分成两路：一路直接传走(残差)，一路进 Norm + Attention
        x = x + self.attn(self.ln1(x), token_positions=token_positions)

        # 步骤 2 ：FFN 子层 (Pre-norm 结构)
        # 两次分流：一路直接传走，一路进 Norm + FFN
        x = x + self.ffn(self.ln2(x))

        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta:float,
                  device=None, 
                  dtype=None,
                  # 新增实验参数
                  use_rms_norm: bool = True,
                  norm_mode: str = "pre",
                  ffn_type: str = "swiglu"):
                  
        super().__init__()

        self.max_seq_len = max_seq_len

        # 1. Token Embedding 层
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        # 2. 堆叠 Transformer Blocks
        # 将实验参数传给每一个 Block
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, max_seq_len, rope_theta,
                device=device, dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        # 3. 最终输出层
        # 如果全局禁用了 Norm, 这里的 Final Norm 也要变成 Identity
        if use_rms_norm:
            self.ln_final = RMSNorm(d_model,device=device, dtype=dtype)
        else:
            self.ln_final = nn.Identity()

        # 4. 一个 Linear 层 映射回 vocab_size 大小
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        b,s = token_ids.shape
        # 准备位置信息用于 RoPE shape:[S] -> [1,S] -> [B,S]
        token_positions = torch.arange(s, device=token_ids.device).unsqueeze(0).expand(b,s)

        # 1. Embedding
        x = self.token_embeddings(token_ids)

        # 2. 逐层通过 Transformer Blocks
        for layer in self.layers:
            x = layer(x, token_positions = token_positions)

        # 3. 最终归一化
        x = self.ln_final(x)

        # 4. 投影到词表空间得到 logits
        return self.lm_head(x)
    

