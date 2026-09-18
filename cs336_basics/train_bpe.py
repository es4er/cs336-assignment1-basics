import os
import time
import psutil
import heapq
from collections import defaultdict, Counter
import regex as re
import json
from .pretokenization import parallel_pretokenize

def train_bpe(
    input_path: str | os.PathLike, # 输入语料文件的路径
    vocab_size: int,           # 目标词表大小（基础字节 + 合并Token + 特殊Token）
    special_tokens: list[str], # 需要保留的特殊 Token 列表
    num_processes: int = 1,
)-> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    """
    训练字节级 BPE(Byte-Pair Encoding)分词器。
    该函数 BPE 算法的核心流程：
    1. 初始化词表为所有可能的字节(0-255)。
    2. 读取输入语料，并根据特殊 Token 进行切分，确保特殊 Token 不参与统计。
    3. 使用 GPT-2 的预分词正则将语料库切分成单词，并统计每个单词的频率。
    4. 迭代进行 “合并” 操作，直到达到目标词表大小。
        - 合并策略：总是选择当前出现频率最高、且在字典序上最大的字节对。
    5. 使用倒排索引优化合并过程中的频率更新，确保速度。
    6. 将合并产生的 Token 加入词表，并最终加入特殊 Token
    
    返回：
    tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        vocab:训练好的词汇表，映射 Token ID → Token 字节序列。
        merges:BPE 合并规则列表，按生成顺序排列。
    """

    # --- 1. 初始化基础词表 ---
    # 词表从 0 到 255 的字节开始，这是BPE的基础单位
    vocab = {i: bytes([i]) for i in range (256)}

    # 计算需要进行的合并次数
    num_merges = vocab_size - 256 - len(special_tokens)

    # --- 2. 预分词
    pretoken_start = time.time()
    raw_counts = parallel_pretokenize(input_path,num_processes,special_tokens,use_parallel=(num_processes > 1))
    pretoken_end = time.time()
    print(f"预分词耗时: {pretoken_end - pretoken_start:.2f} 秒")


    # --- 3. 构建Word representation ----
    # words_list: 存储每个单词的字节列表。
    # counts_list: 存储对应单词的频率。
    words_list = []
    counts_list = []
    for word_tuple, freq in raw_counts.items():
        words_list.append(list(word_tuple))  
        counts_list.append(freq)

    print(f"建立 word 列表完成，共 {len(words_list):,} 个 unique words")

    # defaultdict(int): 一个"带默认初始值"的字典。当你访问一个字典中不存在的键时，他不会报错，而是自动为这个键创建一个默认值 0。
    # stats: pair-> 全局频率
    stats = defaultdict(int)

    # indices: pair-> 包含该pair的word index
    # 倒序索引：这个结构是性能优化的关键，用于快速找到需要更新的单词。
    indices = defaultdict(set)

    # --- 4. 初始化 stats 和 indices ----
    # 遍历所有唯一的单词
    for idx, word in enumerate(words_list):
        freq = counts_list[idx] # 获取该单词的出现频率
        # 遍历单词中所有相邻字节对
        for i in range(len(word) - 1):
            pair = (word[i],word[i+1])
            stats[pair] += freq         # 累加该pair的全局频率
            indices[pair].add(idx)      # 将当前单词的索引加入该 pair 的倒排列表中




    # --- 5. Priority Queue 最小堆实现 ----
    """
    Python heapq 是最小堆
    第一关键字：-frequency
    第二关键字：需要frequency相同的时候选择字典序最大的pair

    因此使用 ReversePair
    """
    class ReversePair:
        def __init__(self,pair):
            self.pair = pair
        def __lt__(self, other):
            return self.pair > other.pair
        
    heap = []
    for pair,freq in stats.items():
        heapq.heappush(heap,(-freq,ReversePair(pair),pair))

    def update_stat(pair,delta):

        # delta 是变化量，正数增加，负数减少
        """
        更新 pair frequency
        同时把新的 frequency 放入 heap
        heap 中允许存在旧数据
        后续通过 lazy deletion 判断是否过期
        """
        if pair is None:
            return

        stats[pair] += delta
        new_freq = stats[pair]

        if new_freq <= 0:
            # 频次 ≤0，说明这个pair已经不存在了，从真实字典删掉
            if pair in stats:
                del stats[pair]
            if pair in indices:
                del indices[pair]
        else:
            heapq.heappush(heap,(-new_freq,ReversePair(pair),pair))




    # --- 6.BPE merge -----
    bpe_start = time.time()

    merges = [] # 用于存储生成的 BPE 合并规则，按顺序记录

    print(
        f"开始 BPE merge，共计划进行 {num_merges:,} 次 merge..."
    )

    for merge_idx in range(num_merges):
        # 6.1 找当前最高频 pair
        best_pair = None
        while heap:
            neg_freq, _, pair = heapq.heappop(heap)
            current_freq = stats.get(pair, 0)
            # heap 中的数据如果和当前 stats 不一致，
            # 说明这是 stale entry，丢弃。
            if current_freq != -neg_freq:
                continue

            best_pair = pair
            break

        # 没有可用 pair
        if best_pair is None:
            break
        best_freq = stats[best_pair]

        # 6.2 记录 merge
        merges.append(best_pair)
        new_token = (best_pair[0] + best_pair[1])

        # 6.3 找到所有包含 best_pair 的 word
        relevant_indices = list(indices.get(best_pair, set()))

        # 6.4 更新这些 word
        for idx in relevant_indices:
            word = words_list[idx]
            freq = counts_list[idx]

            # 先收集该单词中所有相邻 pair（用于后续减少旧pair频率）
            old_pairs_in_word = Counter()
            for j in range(len(word) - 1):
                old_pairs_in_word[(word[j], word[j+1])] += 1

            # 执行合并（一次性合并所有 best_pair 出现位置）
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            # 更新该单词的表示
            words_list[idx] = new_word

            # 收集新单词的所有相邻 pair
            new_pairs_in_word = Counter()
            for j in range(len(new_word) - 1):
                new_pairs_in_word[(new_word[j], new_word[j+1])] += 1

            # 计算需要减少的 pair 集合（旧pair - 新pair）
            for pair, cnt in old_pairs_in_word.items():
                if pair in stats:
                    stats[pair] -= cnt * freq
                    if stats[pair] <= 0:
                        del stats[pair]
                        if pair in indices:
                            # 从该pair的索引集中移除当前单词
                            indices[pair].discard(idx)
                            if not indices[pair]:
                                del indices[pair]
                    else:
                        # 频率减少但未归零，仍然需要从索引中移除当前单词
                        if pair in indices:
                            indices[pair].discard(idx)
                            if not indices[pair]:
                                del indices[pair]
                        # 堆中可能已有旧数据，稍后会通过lazy deletion处理
                        heapq.heappush(heap, (-stats[pair], ReversePair(pair), pair))
                else:
                    # pair 本来就不在 stats 中（可能由于之前合并已被删除），忽略
                    pass

            # 计算需要增加的 pair 集合（新pair - 旧pair）
            for pair, cnt in new_pairs_in_word.items():
                if pair in stats:
                    # 旧pair可能已经被减少，现在增加
                    stats[pair] += cnt * freq
                else:
                    stats[pair] = cnt * freq
                # 更新倒排索引，添加当前单词索引
                if pair not in indices:
                    indices[pair] = set()
                indices[pair].add(idx)
                # 推入堆（允许重复，lazy deletion会处理）
                heapq.heappush(heap, (-stats[pair], ReversePair(pair), pair))

        # 6.5 清理 best_pair
        stats.pop(best_pair,None)
        indices.pop(best_pair,None)

        # 7.输出进度
        current_merge = merge_idx + 1
        if (
            current_merge % 100 == 0
            or current_merge == 1
            or current_merge == num_merges
        ):
            print(
                f"BPE progress: "
                f"{current_merge:,}/{num_merges:,} "
                f"({current_merge / num_merges * 100:.1f}%) | "
                f"pair={best_pair} | "
                f"freq={best_freq:,} | "
                f"remaining_pairs={len(stats):,}"
            )

    # 8. 把 merge token 加入 vocab
    for pair in merges:
        new_id = len(vocab)
        vocab[new_id] = (pair[0]+ pair[1])

    # 9. 加入 special tokens
    for special_token in special_tokens:
        special_bytes = special_token.encode( "utf-8")
        vocab[len(vocab)] = special_bytes

    # 10. 最终检查
    print(
        f"BPE merge 完成："
        f"{len(merges):,} merges"
    )

    print(
        f"最终 vocabulary size："
        f"{len(vocab):,}"
    )

    bpe_end = time.time()
    print(f"BPE 合并耗时: {bpe_end - bpe_start:.2f} 秒")
    return vocab, merges

def bytes_to_unicode():
    """
    创建一个映射，将 0‑255 字节映射为一组可见的 Unicode 字符。
    这是 GPT‑2 源码中的标准做法。
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

def save_tokenizer_files(vocab,merges,out_dir):
    os.makedirs(out_dir, exist_ok= True)

    # 初始化映射表
    byte_encoder = bytes_to_unicode()

    # 词表保存
    # 使用 byte_encoder 将 bytes 转为可见字符串
    json_vocab = {
        str(k): "".join(byte_encoder[b] for b in v)
        for k, v in vocab.items()
    }

    vocab_path = os.path.join(out_dir, "vocab.json")

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(
            json_vocab,
            f,
            ensure_ascii=False,
            indent=2
        )

    merges_path = os.path.join(out_dir, "merges.txt")

    with open(merges_path, "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            s1 = "".join(byte_encoder[b] for b in p1)
            s2 = "".join(byte_encoder[b] for b in p2)

            f.write(f"{s1} {s2}\n")

    print(f"词表已保存：{vocab_path}")
    print(f"合并规则已保存：{merges_path}")

def main():
    start_time = time.time()
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / (1024 ** 3)

    input_path = "../data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000

    special_tokens = ["<|endoftext|>"]
    output_dir = "../data/TinyStoriesV2-GPT4-train"

    print(f"开始训练BPE分词器(目标词表大小：{vocab_size})...")
    print(f"训练数据：{input_path}")
    print("这是正式训练阶段，使用 TinyStories Train 数据集")


    # 单线程运行耗时 28.903s。进程数由 1 提升至 4 时，获得最大加速效果；
    # 继续增大进程数至 6、8、10，耗时稳定在 10‑11s 区间，加速效果趋于饱和，没有进一步显著降低。
    vocab,merges = train_bpe(input_path,vocab_size,special_tokens,num_processes=16)

    save_tokenizer_files(vocab,merges,output_dir)

    print("BPE训练完成!")
    print(f"vocab size: {len(vocab)}")
    print(f"merge count: {len(merges)}")

    end_time = time.time()
    end_memory = process.memory_info().rss / (1024 ** 3)
    print("=" * 50)
    print(f"训练耗时: {(end_time - start_time):.2f} 秒")
    print(f"训练耗时: {(end_time - start_time) / 60:.2f} 分钟")
    print(f"内存使用: {end_memory - start_memory:.2f} GB")
    print(f"峰值内存: {end_memory:.2f} GB")
    print("=" * 50)


if __name__ == "__main__":
    main()
