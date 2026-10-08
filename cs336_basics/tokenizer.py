from __future__ import annotations
import pickle
from collections.abc import Iterable, Iterator
import regex as re
from cs336_basics.train_bpe import merge_word
from cs336_basics.train_bpe import PAT  


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab=dict(vocab)
        bytes_to_id={}
        merge_ranks={}
        for i in range(len(merges)):
            merge_ranks[merges[i]]=i
        self.merge_ranks=merge_ranks
        if not special_tokens:
            self.special_tokens=[]
        else:
            self.special_tokens=special_tokens
        for key, byte in self.vocab.items():
            bytes_to_id[byte]=key
        for special_token in self.special_tokens:
            bytes_special_token=special_token.encode("utf-8")
            if not bytes_special_token in bytes_to_id:
                self.vocab[len(self.vocab)]=bytes_special_token
                bytes_to_id[bytes_special_token]=len(bytes_to_id)
        self.bytes_to_id=bytes_to_id
        self.cache={}


    def decode(self, ids: list[int]) -> str:
        bytes_list=[]
        for id in ids:
            bytes_list.append(self.vocab[id])
        bytes_series=b"".join(bytes_list)
        return bytes_series.decode("utf-8",errors="replace")

    def _apply_merges(self, pretoken: bytes) -> list[bytes]:
        bytes_list=[bytes([b]) for b in pretoken]
        while len(bytes_list)>1:
            pair_list=[]
            for i in range(len(bytes_list)-1):
                pair=bytes_list[i],bytes_list[i+1]
                if tuple(pair) in self.merge_ranks.keys():
                    pair_list.append(pair)
            if not pair_list:
                break
            rank_min_pair=min(pair_list,key=lambda f:self.merge_ranks[tuple(f)])
            new_byte_list=[byte for byte in merge_word(tuple(bytes_list),tuple(rank_min_pair))]
            bytes_list=new_byte_list
        return bytes_list
            

    def _split_on_special(self, text: str) -> list[str]:
        if not text:
            return []
        if not self.special_tokens:
            return [text]
        special_token_reverse=sorted(self.special_tokens,key=len,reverse=True)
        split_list=[re.escape(special_token) for special_token in special_token_reverse]
        split_string="|".join(split_list)
        text_split=re.split("("+split_string+")",text)
        text_split_modified=[str for str in text_split if str]
        return text_split_modified


    def encode(self, text: str) -> list[int]:
        text_list=self._split_on_special(text)
        id_list=[]
        for string in text_list:
            if string in self.special_tokens:
                id_list.append(self.bytes_to_id[string.encode("utf-8")])
                continue
            for pretoken in re.finditer(PAT,string):
                pretoken_bytes=pretoken.group().encode("utf-8")
                if pretoken_bytes not in self.cache:
                    self.cache[pretoken_bytes]=self._apply_merges(pretoken_bytes)
                new_list=self.cache[pretoken_bytes]
                for byte in new_list:
                    id_list.append(self.bytes_to_id[byte])
        return id_list
 

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for string_line in iterable:
            yield from self.encode(string_line)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath,"rb") as f:
            vocab=pickle.load(f)
        with open(merges_filepath,"rb") as f:
            merges=pickle.load(f)
        return Tokenizer(vocab,merges,special_tokens)
