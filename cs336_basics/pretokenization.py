import os
import regex as re
from collections import Counter
from multiprocessing import Pool
from typing import BinaryIO,List,Optional

"""
把大文件切分成N块, 分给多进程分别做预分词、统计词频
但是不能随便按字节切，不能把<|endoftext|>这个特殊token从中间劈开
"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]: # 每个数字是文件字节偏移，分片边界点
    """
    有可能返回实际块数比期望少；多个预估边界搜出来同一个特殊 token 位置，去重之后块变少。
    """
    assert isinstance(split_special_token, bytes), "强制要求分隔符必须是 bytes 二进制"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END) # 把文件指针移动到文件末尾
    file_size = file.tell()
    file.seek(0) # 指针重置回文件开头，后面继续读文件

    chunk_size = file_size // desired_num_chunks

    # 生成初始预估切割点。
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))

def _pretokenize_chunk(args):
    # 处理单个文件快
    chunk_bytes, special_tokens = args
    text = chunk_bytes.decode('utf-8')
    
    """
    For special_tokens:
    在训练时，必须保证特殊 Token 不参与频率统计。
    代码逻辑：
        切割语料: 在开始统计词频之前，利用正则将语料库在特殊 Token 处切开。
        独立统计: 只对切分出来的普通文本片段进行 BPE 统计。
        最后加入: 训练结束后，强制将特殊 Token 加入词表(通常放在最后),确保它们有ID。
    """

    if special_tokens:
        # 在正则中，| 表示“或”，这行代码将多个特殊 token 用 | 连接，形成一个匹配任一 token 的正则模式。
        special_regex = "|".join(re.escape(t) for t in special_tokens)
        # 使用 re.split 进行分割。关键是使用捕获组 `(...)`，这样特殊 Token 本身也会被保留在结果列表中。
        parts = re.split(f"({special_regex})", text)
        # 过滤掉从 parts 中提取出的特殊 Token 本身，只保留用于 BPE 训练的普通文本片段。
        # text = "Hello World World<|endoftext|>Hello happy happy<|endoftext|>!"
        # train_segments = ['Hello World World', 'Hello happy happy', '!']
        train_segments = [p for p in parts if p not in special_tokens]
    else:
        # 如果没有特殊 Token，直接使用整个语料。
        train_segments = [text]


    # 使用 GPT‑2 的 BPE 预分词正则表达式。
    # GPT‑2 正则表达式的作用是执行“预分词（Pre‑tokenization）”。它的规则是：
    #   (1)不允许跨越类型合并：比如它会把字母和标点符号分开。
    #   (2)保护空格：它通常会把单词前面的空格和单词连在一起，作为一个整体。
    # text = "Hello World test! ..."
    # 分割后 words = ['Hello', ' World', 'test', '!', ' ...']
    gpt2_pat = re.compile(r"""(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    # raw_counts: 存储每个“单词”（预分词后的结果）及其出现频率。
    # 单词被表示为字节元组，例如 "hello" -> (b'h', b'e', b'l', b'l', b'o')
    raw_counts = Counter()
    print("开始进行 pre-tokenization...")
    for segment in train_segments:
        # 对每个语料片段应用预分词正则，找到所有“单词”
        words = gpt2_pat.findall(segment)
        for word in words:
            """
                对于 "Hi":
                    word.encode("utf-8") 得到 b'Hi'。
                    for b in b'Hi' 会遍历出整数 72 和 105。
                    bytes([b]) 把整数变回单字节对象: b'H' 和 b'i'。
                    最终组成元组: (b'H', b'i')。
                为什么必须是元组(tuple)?
                    因为 Counter 的键(key)必须是不可变的。list 不能做键，而 tuple 可以。
    
                举例:
                raw_counts = {
                    (b'H', b'i'): 50,
                    (b' ', b't', b'h', b'e', b'r', b'e'): 100,
                    (b'!'): 50,
                    (b'\xe4', b'\xbd', b'\xa0', b'\xe5', b'\xa5', b'\xbd'):20, # 你好
                }
            """
            raw_counts[tuple(bytes([b]) for b in word.encode("utf-8"))] += 1
    
    return raw_counts

def parallel_pretokenize(
        file_path: str,
        num_processes: int,
        special_tokens: List[str],
        use_parallel: bool = True
)->Counter:
    """
    并行预分词,返回词频统计
    """
    if not use_parallel or num_processes <= 1:
        # 串行回退模式
        print("使用串行模式预分词...")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        if special_tokens:
            special_regex = "|".join(re.escape(t) for t in special_tokens)
            parts = re.split(f"({special_regex})", text)
            train_segments = [p for p in parts if p not in special_tokens]
        else:
            train_segments = [text]
        
        gpt2_pat = re.compile(r"""(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        raw_counts = Counter()
        for segment in train_segments:
            words = gpt2_pat.findall(segment)
            for word in words:
                raw_counts[tuple(bytes([b]) for b in word.encode("utf-8"))] += 1
        
        return raw_counts
    
    # 并行模式
    print(f"使用 {num_processes} 个进程并行预分词...")
    split_token = special_tokens[0].encode('utf-8') if special_tokens else None
    
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_token)
    
    chunks = []
    with open(file_path, "rb") as f:
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_data = f.read(end - start)
            chunks.append((chunk_data, special_tokens))
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(_pretokenize_chunk, chunks)
    
    total = Counter()
    for c in results:
        total.update(c)
    
    print(f"预分词完成，共 {len(total):,} 个唯一单词")
    return total


