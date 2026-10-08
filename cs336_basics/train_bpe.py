import os
from multiprocessing import Pool
import regex as re  
from typing import BinaryIO
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""



def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab={}
    for i in range(256):
        vocab[i]=bytes([i])
    for special_token in special_tokens:
        vocab[len(vocab)]=special_token.encode("utf-8")
    return vocab

def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    split_text=[]
    normalized_special_tokens=[]
    special_token_string=""
    if not special_tokens:
        split_text.append(text)
        return split_text
    normalized_special_tokens=[re.escape(special_token) for special_token in special_tokens]
    special_token_string="|".join(normalized_special_tokens)
    split_text=re.split(special_token_string,text)
    return split_text

def pretokenize_and_count(chunks: list[str]) -> dict[tuple[bytes, ...], int]:
    pretokenized_dict={}
    for chunk in chunks:
        for pretoken in re.finditer(PAT,chunk):
            pretokenized_list=[num for num in pretoken.group().encode("utf-8")]
            new_key=[bytes([i]) for i in pretokenized_list]
            pretokenized_dict[tuple(new_key)]=pretokenized_dict.get(tuple(new_key),0)+1
    return pretokenized_dict


def count_pairs(word_count_copy: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    pair_counts={}
    for word, count in word_count_copy.items():
        for i in range(0,len(word)-1):
            pair_counts[tuple([word[i],word[i+1]])]=pair_counts.get(tuple([word[i],word[i+1]]),0)+count
    return pair_counts

def pick_best_pair(pair_counts: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
    return max(pair_counts,key=lambda k:(pair_counts[k],k))

def merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    i=0
    new_word=[]
    pair_add=pair[0]+pair[1]
    while i<len(word)-1:
        if word[i]==pair[0] and word[i+1]==pair[1]:
            new_word.append(pair_add)
            i=i+2
        else:
            new_word.append(word[i])
            i=i+1
    if i==len(word)-1:
        new_word.append(word[i])
    return tuple(new_word)

def run_merges_slow(word_count_copy: dict[tuple[bytes, ...], int], num_merges: int) -> list[tuple[bytes, bytes]]:
    merges=[]
    for _ in range(num_merges):
        new_dict={}
        pair_counts=count_pairs(word_count_copy)
        if not pair_counts:
            break
        pair=pick_best_pair(pair_counts)
        for word in word_count_copy:
            new_word=merge_word(word,pair)
            new_dict[new_word]=word_count_copy[word]
        merges.append(pair)
        word_count_copy=new_dict
    return merges




def build_pair_index(word_count_copy: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]:
    pair_word_dict={}
    for key in word_count_copy:
        for i in range(len(key)-1):
            pair=tuple([key[i],key[i+1]])
            if pair not in pair_word_dict:
                word=set()
                pair_word_dict[pair]=word
            pair_word_dict[pair].add(key)
    return pair_word_dict
   
def remove_word(
    word: tuple[bytes, ...],
    count: int,
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
) -> None:
   
    for i in range(len(word)-1):
        pair=tuple([word[i],word[i+1]])
        pair_counts[pair]=pair_counts[pair]-count
        if (pair_counts[pair]==0):
            del pair_counts[pair]
        pair_to_words[pair].discard(word)

def add_word(
    word: tuple[bytes, ...],
    count: int,
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
) -> None:
    for i in range(len(word)-1):
        pair=tuple([word[i],word[i+1]])
        pair_counts[pair]=pair_counts.get(pair,0)+count
        if pair not in pair_to_words:
            pair_to_words[pair]=set()
        pair_to_words[pair].add(word)

def run_merges_fast(word_counts: dict[tuple[bytes, ...], int], num_merges: int) -> list[tuple[bytes, bytes]]:
    merges=[]
    word_count_copy=dict(word_counts)
    pair_counts=count_pairs(word_count_copy)
    pair_to_word_dict=build_pair_index(word_count_copy)
    for _ in range(num_merges):
        if not pair_counts:
            break
        pair=pick_best_pair(pair_counts)
        merges.append(pair)
        words_set=pair_to_word_dict[pair]
        iteration_word=list(words_set)
        for word in iteration_word:
            count=word_count_copy.pop(word)
            remove_word(word,count,pair_counts,pair_to_word_dict)
            new_word=merge_word(word,pair)
            add_word(new_word,count,pair_counts,pair_to_word_dict)
            word_count_copy[new_word]=word_count_copy.get(new_word,0)+count
    return merges


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))
def count_chunk(
    input_path: str | os.PathLike,
    start: int,
    end: int,
    special_tokens: list[str],
) -> dict[tuple[bytes, ...], int]:
    with open(input_path,"rb") as f:
        f.seek(start)
        input_read=f.read(end-start).decode("utf-8",errors="ignore")
    chunks=split_on_special_tokens(input_read,special_tokens)
    small_dic=pretokenize_and_count(chunks)
    return small_dic

def merge_counts(list_of_counts: list[dict[tuple[bytes, ...], int]]) -> dict[tuple[bytes, ...], int]:
    big_dic={}
    for small_dic in list_of_counts:
        for key in small_dic:
            big_dic[key]=big_dic.get(key,0)+small_dic[key]
    return big_dic
def pretokenize_parallel(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int,
) -> dict[tuple[bytes, ...], int]:

    with open(input_path,"rb") as f:
        edging_point=find_chunk_boundaries(f,num_processes,special_tokens[0].encode("utf-8"))
    mission_list=[]
    for start, end in zip(edging_point[:-1],edging_point[1:]):
        mission_list.append(tuple([input_path,start,end,special_tokens]))
    with Pool(num_processes) as p:
        list_of_counts=p.starmap(count_chunk,mission_list)
    return merge_counts(list_of_counts)

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    word_count_copy=pretokenize_parallel(input_path,special_tokens,num_processes=os.cpu_count())
    vocab=init_vocab(special_tokens)
    num_merges=vocab_size-len(vocab)
    merges=run_merges_fast(word_count_copy,num_merges)
    for x,y in merges:
        vocab[len(vocab)]=x+y
    return vocab,merges