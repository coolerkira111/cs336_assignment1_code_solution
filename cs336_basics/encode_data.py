import sys
import time
import numpy as np
from cs336_basics.tokenizer import Tokenizer

# 用法: uv run python -m cs336_basics.encode_data <输入 txt> <输出 npy>
src, dst = sys.argv[1], sys.argv[2]
tok = Tokenizer.from_files("tinystories_train_vocab.pkl", "tinystories_train_merges.pkl", ["<|endoftext|>"])

chunks = []   # 每个元素是一个 uint16 小数组
buf = []      # 普通 list,攒满 100 万个 id 就转成 uint16 存进 chunks
t0 = time.perf_counter()
with open(src, encoding="utf-8") as f:          # 文件对象逐行读,不会一次读进整个文件
    for token_id in tok.encode_iterable(f):
        buf.append(token_id)
        if len(buf) >= 1_000_000:
            chunks.append(np.array(buf, dtype=np.uint16))
            buf = []
chunks.append(np.array(buf, dtype=np.uint16))   # 最后剩下的不满 100 万个

ids = np.concatenate(chunks)                    # 把所有小数组接成一个
np.save(dst, ids)
print(f"{len(ids)} tokens, {time.perf_counter() - t0:.1f}s, saved to {dst}")
