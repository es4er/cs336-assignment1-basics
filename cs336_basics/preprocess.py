import os
import json
import time
import numpy as np
from typing import List, Dict, Iterable
from .tokenizer import BPETokenizer


def bytes_to_unicode():
    """
    返回一个字节到可见 Unicode 字符的映射字典。
    该映射确保所有 256 个字节值都有对应的可见字符
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def load_trained_tokenizer(vocab_path: str, merges_path: str, special_tokens: List[str]):
    """
    从磁盘加载训练好的分词器，处理 byte_to_unicode 反向映射
    """
    print(f"正在从 {os.path.dirname(vocab_path)} 加载分词器...")
    start_time = time.time()
    # 1. 建立反向映射表
    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    # 2. 加载并还原词表
    print(f"  - 加载词表: {vocab_path}")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_raw = json.load(f)
        # 将可见的 Unicode 字符串还原为原始 bytes
        vocab = {
            int(k): bytes([byte_decoder[c] for c in v])
            for k, v in vocab_raw.items()
        }
    print(f"  - 词表大小: {len(vocab):,}")

    # 3. 加载并还原合并规则
    print(f"  - 加载合并规则: {merges_path}")
    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            # 使用 rsplit 确保在 token 本身包含空格的情况下（虽然映射后不应该有空格）依然稳健
            parts = line.split(' ')
            if len(parts) == 2:
                # 还原 p1, p2 为原始 bytes
                p1 = bytes([byte_decoder[c] for c in parts[0]])
                p2 = bytes([byte_decoder[c] for c in parts[1]])
                merges.append((p1, p2))

    print(f"  - 合并规则数: {len(merges):,}")

    elapsed = time.time() - start_time
    print(f" 分词器加载完成，耗时 {elapsed:.2f} 秒...")
    return BPETokenizer(vocab, merges, special_tokens)


def process_corpus(input_txt: str, output_bin: str, tokenizer: BPETokenizer, chunk_size_mb: int = 100):

    start_time = time.time()

    # 1. 内部生成器：负责按块从硬盘读文本
    def file_chunk_generator(file_path, size):
        with open(file_path, "r", encoding="utf-8") as f:
            while True:
                chunk = f.read(size)
                if not chunk:
                    break
                yield chunk

    # 2. 检查与准备
    if not os.path.exists(input_txt):
        raise FileNotFoundError(f"找不到语料文件: {input_txt}")

    file_size = os.path.getsize(input_txt)
    file_size_mb = file_size / (1024 * 1024)

    chunk_size = 1024 * 1024 * chunk_size_mb

    if os.path.exists(output_bin):
        os.remove(output_bin)

    print(f"使用 encode_iterable 开始流式处理...")

    # 3. 核心流式逻辑
    # 创建文本块生成器
    chunks = file_chunk_generator(input_txt, chunk_size)
    # 丢进 encode_iterable, 得到一个“不停吐出 ID”的生成器
    token_stream = tokenizer.encode_iterable(chunks)

    total_tokens = 0

    # 为了高效写入硬盘，我们依然需要一个小缓存 (Buffer)
    # 每积攒 500 万个 Token 写入一次硬盘
    write_batch_size = 5_000_000
    token_buffer = []

    # 进度显示
    last_log_time = time.time()
    log_interval = 5  # 每 5 秒显示一次进度
    bytes_processed = 0
    print("开始编码")


    with open(output_bin, "ab") as f_out:
        for token_id in token_stream:
            token_buffer.append(token_id)

            if len(token_buffer) >= write_batch_size:
                np_ids = np.array(token_buffer, dtype=np.uint16)
                f_out.write(np_ids.tobytes())
                total_tokens += len(token_buffer)
                token_buffer = []

                # 显示进度（每 5 秒或每 1% 显示一次）
                current_time = time.time()
                if current_time - last_log_time >= log_interval:
                    progress_pct = (bytes_processed / file_size) * 100
                    elapsed = current_time - start_time
                    speed = bytes_processed / elapsed / (1024 * 1024)  # MB/s
                    print(f"    进度: {progress_pct:.1f}% | "
                          f"已处理: {bytes_processed / (1024*1024):.1f} MB | "
                          f"已编码: {total_tokens:,} tokens | "
                          f"速度: {speed:.1f} MB/s")
                    last_log_time = current_time

        # 处理最后剩余的 buffer
        if token_buffer:
            np_ids = np.array(token_buffer, dtype=np.uint16)
            f_out.write(np_ids.tobytes())
            total_tokens += len(token_buffer)

    elapsed = time.time() - start_time
    output_size = os.path.getsize(output_bin) / (1024 * 1024)

    print(f"处理完成!")
    print(f"  - 总 Token 数: {total_tokens:,}")
    print(f"  - 输出文件大小: {output_size:.2f} MB")
    print(f"  - 总耗时: {elapsed:.2f} 秒 ({elapsed/60:.2f} 分钟)")
    print(f"  - 平均速度: {total_tokens/elapsed:.0f} tokens/秒")
    print(f"  - 压缩比: {file_size_mb / output_size:.2f}x (原始文件/编码文件)")


def main():
    # --- 配置区 ---
    # 根据你的训练结果修改路径
    BASE_DIR = "../data/TinyStoriesV2-GPT4-train"
    input_file = "../data/TinyStoriesV2-GPT4-train.txt"  # 待处理的原始语料
    output_file = "../data/TinyStoriesV2-GPT4-train.bin"  # 输出的二进制文件

    vocab_json = os.path.join(BASE_DIR, "vocab.json")
    merges_txt = os.path.join(BASE_DIR, "merges.txt")
    special_tokens = ["<|endoftext|>"]

    # 1. 加载分词器
    tokenizer = load_trained_tokenizer(vocab_json, merges_txt, special_tokens)

    # 2. 执行数据清洗与预处理
    process_corpus(input_file, output_file, tokenizer)


if __name__ == "__main__":
    main()
