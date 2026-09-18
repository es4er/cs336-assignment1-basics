import regex as re
from collections.abc import Iterable

"""
For special_tokens:
    推理/编码阶段(Tokenizer.encode)
        在模型使用分词器将文本转成 ID 时,必须优先匹配特殊 Token
    代码逻辑
        正则匹配: 构建一个包含所有特殊 Token 的正则表达式
        优先级: 先扫描文本, 一旦发现特殊 Token , 直接将其转为对应的 ID
        普通处理: 特殊 Token 之间的文本, 再走正常的 GPT-2 预分词和 BPE 合并流程
"""

class BPETokenizer:
    """
    字节级 BPE (Byte-Pair Encoding) 分词器实现

    该分词器将任意字符编码为整数 ID 序列, 并能将 ID 序列还原
    """
    def __init__(self,vocab: dict[int,bytes],merges: list[tuple[bytes,bytes]],special_tokens: list[str] | None = None):
        """
        初始化分词器。
        参数：
        vocab: 词汇表，建立整数 ID 到 字节块(bytes) 的映射。
        merges: 合并规则列表。列表中的每一项是一个二元组 (bytes_a, bytes_b)，
            表示在训练过程中 bytes_a 和 bytes_b 被合并的顺序。
        special_tokens: 特殊标记列表 (如 <|endoftext|>)，这些标记不会被 BPE 规则拆分。
        """

        # 1. 构建双向映射，方便查表
        self.vocab = vocab
        self.id_to_byte = vocab
        self.byte_to_id = {v:k for k, v in vocab.items()}

        # 2. 将合并规则转为 Rank 字典
        # BPE 编码时，必须先优先应用在训练阶段较早出现的合并规则
        # 字典结构为：{(byte_a,byte_b):顺序索引}
        self.merges = {pair:i for i,pair in enumerate(merges)}

        self.special_tokens = special_tokens or []

        # 3. 构建特殊 Token 的正则表达式
        if self.special_tokens:
        # 关键: 必须按照长度从长到短排序 (reverse=True)。
        # 这样正则引擎会优先匹配最长的特殊标记, 防止重叠标记 (如 <|a|><|b|>) 被错误拆分。
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            # 使用 re.escape 确保标记中的特殊字符 (如 | 或 [ ) 被当作普通字符处理
            special_pattern = "|".join(re.escape(t) for t in sorted_special)
            self.special_regex = re.compile(special_pattern)
        else:
            self.special_regex = None

        # 4. GPT‑2 官方预分词正则表达式。
        # 它的作用是在应用 BPE 合并前, 先将文本切分成单词、标点、数字等逻辑块。
        # 这样做是为了防止 BPE 规则跨越单词或标点 (例如: 防止将 "dog" 的末尾和 "." 合并)。
        self.gpt2_pat = re.compile(r"""(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def encode(self, text: str) -> list[int]:
        """
        将输入的原始字符串编码为整数 ID 列表。

        该方法的核心逻辑是：
        1. 作为一个“协调者”，它负责处理文本中的“特殊标记(Special Tokens)”和“普通文本”。
        2. 特殊标记 (如 <|endoftext|>) 被视为原子，直接映射为 ID，不参与 BPE 的拆分和合并。
        3. 普通文本片段则被交给底层逻辑执行预分词和 BPE 算法。

        参数：
            text: 需要编码的原始字符串 (例如 "Hello<|end|>World")。

        返回：
            list[int]: 编码后的整数 ID 序列。
        """

        # --- 步骤 1: 边界情况检查 ---
        # 如果输入是空字符串或 None，直接返回空列表。
        # 这是为了防止后续逻辑在处理空文本时产生错误。
        if not text:
            return []

        # --- 步骤 2: 情况 A - 快速路径（Fast Path） ---
        # 如果我们在初始化时没有定义任何特殊标记（或者特殊标记列表为空），
        # 那么整个文本都可以被视为一段连续的“普通文本”。
        # 我们直接调用内部方法 _encode_text_segment 进行 BPE 处理并返回结果。
        if not self.special_regex:
            return self._encode_text_segment(text)

        # --- 步骤 3: 情况 B - 处理含有特殊标记的复杂文本 ---
        # 此时文本中可能混有普通文字和特殊标记，我们需要像“剪刀”一样把它们切开。
        tokens = []

        # last_pos 用于记录上一次匹配结束的位置，帮助我们定位“特殊标记”之间的“缝隙”。
        last_pos = 0

        # 使用 finditer 遍历文本中所有符合特殊标记模式的匹配项。
        # finditer 的好处是它提供了 match.start() 和 match.end(),
        # 这让我们能够精确地知道特殊标记在哪里开始，在哪里结束。
        for match in self.special_regex.finditer(text):

            # 3.1 提取并处理“前置普通文本”
            # 这里的区间是 [last_pos, match.start())。
            # 这段文本是夹在两个特殊标记之间（或者开头到第一个特殊标记之间）的普通文字。
            pre_text = text[last_pos:match.start()]

            # 如果这两个标记之间确实有文字（长度 > 0）
            if pre_text:
                tokens.extend(self._encode_text_segment(pre_text))

            # 3.2 处理“当前特殊标记”
            # match.group() 拿到的就是被识别出来的特殊标记字符串（如 "<|endoftext|>"）。
            special_tok = match.group()
            # 我们直接将其编码为 UTF‑8 字节，然后在词表中查找其 ID。
            tokens.append(self.byte_to_id[special_tok.encode("utf-8")])

            # 3.3 更新游标
            # 将游标移动到当前匹配项的末尾，为寻找下一个片段做准备。
            last_pos = match.end()

        # --- 步骤 4: 处理“收尾文本” ---
        # 如果最后一个特殊标记后面还有文字（例如 "Hello<|end|>World" 中的 "World"），
        # 或者整个文本根本没有特殊标记匹配（虽然逻辑上 Case A 已处理，但这里是双重保险），
        # 我们需要处理从 last_pos 到字符串末尾的所有剩余字符。
        remaining_text = text[last_pos:]
        if remaining_text:
            # 剩余部分同样作为普通文本片段进行 BPE 编码。
            tokens.extend(self._encode_text_segment(remaining_text))

        # 返回拼接好的所有 ID 列表
        return tokens

    def _encode_text_segment(self, text: str) -> list[int]:
            """
            内部核心函数: 对不含特殊 Token 的纯文本片段应用 BPE 合并逻辑。
            """
            ids = []
            # 使用 GPT‑2 正则进行预分词, 将文本拆成单词/标点符号块
            # 例如: "Hello world!" -> ["Hello", " world", "!"]
            pre_tokens = self.gpt2_pat.findall(text)
    
            for p_tok in pre_tokens:
                # 第一步: 将当前片段转为字节序列, 并将每个字节看作一个独立的"部分 (Part)"
                # 例如: "Hello" -> [b'H', b'e', b'l', b'l', b'o']
                byte_parts = [bytes([b]) for b in p_tok.encode("utf-8")]
    
                # 第二步: 反复执行合并, 直到没有符合条件的合并规则为止
                while len(byte_parts) >= 2:
                    # 在当前序列的所有相邻对中, 寻找合并优先级最高 (Rank 最小) 的一对, 即按照构造merge时添加pair的顺序进行合并
                    best_pair = None
                    min_rank = float('inf')
    
                    for i in range(len(byte_parts) - 1):
                        pair = (byte_parts[i], byte_parts[i+1])
                        if pair in self.merges:
                            rank = self.merges[pair]
                            if rank < min_rank:
                                min_rank = rank
                                best_pair = pair
    
                    # 如果找不到任何可以合并的规则, 退出当前片段的合并过程
                    if best_pair is None:
                        break
    
                    # 第三步: 执行合并操作。
                    # 遍历当前序列, 将所有出现的 best_pair 替换成合并后的长字节块。
                    new_byte_parts = []
                    i = 0
                    # [b'H', b'e', b'l', b'l', b'o'] -> [b'He', b'l', b'l', b'o']
                    while i < len(byte_parts):
                        # 如果当前两个部分匹配最高优规则
                        if i < len(byte_parts) - 1 and (byte_parts[i], byte_parts[i+1]) == best_pair:
                            new_byte_parts.append(best_pair[0] + best_pair[1])
                            i += 2  # 跳过下一项, 因为已经合并了
                        else:
                            new_byte_parts.append(byte_parts[i])
                            i += 1
    
                    byte_parts = new_byte_parts  # 更新序列, 进入下一轮 while 循环
    
                # 第四步: 将合并到极限后的所有字节块转换为词表中的 ID
                for part in byte_parts:
                    ids.append(self.byte_to_id[part])
    
            return ids

    def decode(self, ids: list[int]) -> str:
        """
        将 ID 列表解码为原始字符串。
        """
        # 1. 根据 ID 查表找回字节块
        byte_segments = [self.id_to_byte[i] for i in ids]

        # 2. 将所有字节块按顺序拼接成一个完整的字节流
        full_bytes = b"".join(byte_segments)

        # 3. 将字节流解码为 UTF‑8 字符串。
        # 使用 errors="replace" 非常关键: 因为 BPE 可能会生成不完整的字节序列
        # (例如 3 字节的中文字符只产生了一部分), 此时不报错而是插入替换符。
        return full_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        内存高效的迭代编码器。

        参数:
            iterable: 一个可迭代的字符串对象（例如文件句柄）。
        返回:
            一个生成器，逐个产出编码后的 ID。用于处理无法一次性读入内存的大文件。
        """
        for chunk in iterable:
            # 对每一块文本进行编码，并通过 yield 吐出结果
            yield from self.encode(chunk)