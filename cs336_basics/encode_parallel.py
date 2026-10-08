# 多进程版的 encode_data.py:把文件按 <|endoftext|> 切成很多块,每个进程编码一块,最后按顺序接起来
# 用法: python -m cs336_basics.encode_parallel <输入 txt> <输出 npy> <vocab.pkl> <merges.pkl>
#       进程数默认是 CPU 核数,想少用几个核就在前面加 ENC_PROCS=8
import io
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe import find_chunk_boundaries

tok = None   # 每个进程自己的 tokenizer(带自己的 cache)


def init_worker(vocab_path, merges_path):
    global tok
    tok = Tokenizer.from_files(vocab_path, merges_path, ["<|endoftext|>"])


def encode_chunk(task):
    path, start, end = task
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    chunks, buf = [], []
    for token_id in tok.encode_iterable(io.StringIO(text)):   # 和 encode_data.py 一样逐行编码
        buf.append(token_id)
        if len(buf) >= 1_000_000:
            chunks.append(np.array(buf, dtype=np.uint16))
            buf = []
    chunks.append(np.array(buf, dtype=np.uint16))
    return np.concatenate(chunks)


if __name__ == "__main__":
    src, dst, vocab_path, merges_path = sys.argv[1:5]
    num_processes = int(os.environ.get("ENC_PROCS", min(os.cpu_count(), 12)))
    t0 = time.perf_counter()
    with open(src, "rb") as f:
        bounds = find_chunk_boundaries(f, 256, b"<|endoftext|>")
    tasks = [(src, s, e) for s, e in zip(bounds[:-1], bounds[1:])]
    parts = []
    with Pool(num_processes, initializer=init_worker, initargs=(vocab_path, merges_path)) as pool:
        for i, part in enumerate(pool.imap(encode_chunk, tasks)):   # imap 保证结果按块的顺序回来
            parts.append(part)
            print(f"chunk {i + 1}/{len(tasks)}  {time.perf_counter() - t0:.0f}s", flush=True)
    ids = np.concatenate(parts)
    np.save(dst, ids)
    print(f"{len(ids)} tokens, {time.perf_counter() - t0:.1f}s, saved to {dst}")
