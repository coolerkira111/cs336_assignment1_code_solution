from cs336_basics.train_bpe import train_bpe
import time
import pickle
if __name__ =="__main__":
    t0=time.perf_counter()
    vocab, merges=train_bpe("data/TinyStoriesV2-GPT4-train.txt",10000,["<|endoftext|>"])
    t1=time.perf_counter()
    with open("tinystories_train_vocab.pkl","wb") as f:
        pickle.dump(vocab,f)
    with open("tinystories_train_merges.pkl","wb") as f:
        pickle.dump(merges,f)
    timecost=t1-t0
    print(f"timing={timecost:.2f}s")