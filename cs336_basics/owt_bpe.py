# 训练 OWT 的 BPE。用的是你 train_bpe.py 里的函数,只改了两处,结果和 train_bpe() 完全一样:
#   1. 文件切成很多小块(默认 400 块),一块一块统计再加起来,内存不会爆
#   2. 每次找出现最多的 pair 用"堆"(heapq),不用每次扫一遍所有 pair,快很多
# 用法: python -m cs336_basics.owt_bpe data/owt_train.txt 32000 owt
#       会生成 owt_vocab.pkl 和 owt_merges.pkl
import heapq
import os
import pickle
import sys
import time
from multiprocessing import Pool

from cs336_basics.train_bpe import (add_word, build_pair_index, count_chunk, count_pairs,
                                    find_chunk_boundaries, init_vocab, merge_word, remove_word)


class Item:
    # 堆里的一个元素。heapq 每次弹出"最小"的,所以这里把比较反过来:count 大的、pair 大的算"小"
    __slots__ = ("count", "pair")

    def __init__(self, count, pair):
        self.count = count
        self.pair = pair

    def __lt__(self, other):
        return (self.count, self.pair) > (other.count, other.pair)


def count_task(task):
    return count_chunk(*task)


def pretokenize(path, special_tokens, num_chunks, num_processes):
    with open(path, "rb") as f:
        bounds = find_chunk_boundaries(f, num_chunks, special_tokens[0].encode("utf-8"))
    tasks = [(path, s, e, special_tokens) for s, e in zip(bounds[:-1], bounds[1:])]
    counts = {}
    with Pool(num_processes) as pool:
        for i, small in enumerate(pool.imap_unordered(count_task, tasks)):
            for word, c in small.items():
                counts[word] = counts.get(word, 0) + c
            print(f"pretokenize {i + 1}/{len(tasks)}", flush=True)
    return counts


def run_merges_heap(word_counts, num_merges):
    words = dict(word_counts)
    pair_counts = count_pairs(words)
    pair_to_words = build_pair_index(words)
    heap = [Item(c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    merges = []
    t0 = time.perf_counter()
    while len(merges) < num_merges:
        # 堆里可能有过期的旧计数,和当前计数对不上的直接扔掉
        while heap and pair_counts.get(heap[0].pair, 0) != heap[0].count:
            heapq.heappop(heap)
        if not heap:
            break
        pair = heapq.heappop(heap).pair
        merges.append(pair)
        touched = set()
        for word in list(pair_to_words[pair]):
            count = words.pop(word)
            remove_word(word, count, pair_counts, pair_to_words)
            new_word = merge_word(word, pair)
            add_word(new_word, count, pair_counts, pair_to_words)
            words[new_word] = words.get(new_word, 0) + count
            touched.update(zip(word, word[1:]))
            touched.update(zip(new_word, new_word[1:]))
        for p in touched:
            if p in pair_counts:
                heapq.heappush(heap, Item(pair_counts[p], p))
        if len(merges) % 1000 == 0:
            print(f"merge {len(merges)}/{num_merges}  {time.perf_counter() - t0:.0f}s", flush=True)
    return merges


if __name__ == "__main__":
    path, vocab_size, prefix = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    num_processes = int(os.environ.get("BPE_PROCS", min(os.cpu_count(), 12)))
    special_tokens = ["<|endoftext|>"]
    t0 = time.perf_counter()
    counts = pretokenize(path, special_tokens, num_chunks=400, num_processes=num_processes)
    print(f"pretokenize done: {len(counts)} distinct words, {time.perf_counter() - t0:.0f}s", flush=True)
    vocab = init_vocab(special_tokens)
    merges = run_merges_heap(counts, vocab_size - len(vocab))
    for x, y in merges:
        vocab[len(vocab)] = x + y
    with open(f"{prefix}_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(f"{prefix}_merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    print(f"done: vocab {len(vocab)}, total {time.perf_counter() - t0:.0f}s")
