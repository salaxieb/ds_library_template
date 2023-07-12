import re
from collections import Counter, defaultdict
from typing import List, Tuple
from tqdm import trange

from pathlib import Path


class Tokenizer:
    def __init__(self, vocab_size: int = 0):
        self.vocab_size = vocab_size
        self.re_token = re.compile(r"[^a-zA-Z\d\s]|[a-zA-Z\d]+|\n")
        self.token2id = {}
        self.id2token = {}
        # EOS or PAD token must be number 0, because it will be taken into consideration in loss function
        self.special_tokens = {
            0: "<EOS>",
            1: "<UNK>",
            2: "<NL>",
        }
        # cache for token, which can't encoded to single token
        # and tricky splitting must be selected
        self.encoding_cache = {}

    def fit(self, corpus_path: Path, save_path: Path):
        text_files = corpus_path.glob("*.txt")
        bytes_count = self.bytes_counts_init(text_files)
        vocab = self.init_vocab(bytes_count)

        for _ in trange(len(vocab), self.vocab_size, 1, desc="tokenizer training"):
            bp_counts = self.count_byte_pairs(bytes_count)
            most_common_byte_pair = max(bp_counts, key=lambda key: bp_counts[key])
            # remove ## in the beginning of second symbol
            joined = f"{most_common_byte_pair[0]}{most_common_byte_pair[1][2:]}"
            # print("most_common_byte_pair", most_common_byte_pair)
            vocab.append(joined)
            bytes_count = self.combine_byte_pair(
                most_common_byte_pair, bytes_count, joined
            )

        with open(save_path, "w") as f:
            for token in vocab:
                f.write(f"{token}\n")

        self.token2id = {token: i for i, token in enumerate(vocab)}
        self.id2token = {i: token for token, i in self.token2id.items()}
        self.vocab_size = len(self.token2id)

    def from_pretrained(self, vocab_path):
        with open(vocab_path, "r") as f:
            vocab = f.read().splitlines()
            for i, token in enumerate(vocab):
                self.token2id[token] = i
                self.id2token[i] = token
        # returning back <NL> to \n
        new_line_index = self.token2id["<NL>"]
        self.token2id["\n"] = new_line_index
        self.id2token[new_line_index] = "\n"
        del self.token2id["<NL>"]
        self.vocab_size = len(self.token2id)
        assert self.token2id["<EOS>"] == 0, "loss requirement"
        print("loaded pretrained tokeinzer, vocab size:", self.vocab_size)
        return self

    def encode_token(self, token) -> Tuple[List[int], int]:
        if token in self.token2id:
            return [self.token2id[token]], 1

        if token in self.encoding_cache:
            return self.encoding_cache[token]
        # if we have single char and it not in vocab, use <unk> and set weight to 3
        if len(token) == 1 or (token.startswith("##") and len(token) == 3):
            return [self.token2id["<UNK>"]], 3
        # try to split token to 2 sub parts
        starting_index = 1
        if token.startswith("##"):
            starting_index = 3

        left_ids, score_left = self.encode_token(token[:starting_index])
        right_ids, score_right = self.encode_token(f"##{token[starting_index:]}")
        min_ids = left_ids + right_ids
        min_score = score_left + score_right
        for i in range(starting_index + 1, len(token)):
            left_ids, score_left = self.encode_token(token[:starting_index])
            right_ids, score_right = self.encode_token(f"##{token[starting_index:]}")
            ids = left_ids + right_ids
            score = score_left + score_right
            if score < min_score:
                min_ids = ids
                min_score = score
        self.encoding_cache[token] = (min_ids, min_score)
        return (min_ids, min_score)

    def encode(self, string: str) -> List[int]:
        resulting = []
        tokens = self.re_token.findall(string)
        for token in tokens:
            resulting += self.encode_token(token)[0]
        return resulting

    def decode(self, ids: List[int]) -> str:
        resulting = ""
        for i in ids:
            token = self.id2token[i]
            if token.startswith("##"):
                resulting += token[2:]
            else:
                if resulting != "":
                    resulting += " "
                resulting += token
        return resulting

    def init_vocab(self, bytes_count):
        vocab = set()
        for bytess, _ in bytes_count:
            vocab = vocab.union(bytess)
        vocab = list(vocab)
        for index, token in self.special_tokens.items():
            vocab.insert(index, token)
        return vocab

    def bytes_counts_init(self, text_paths: List[Path]) -> List[Tuple[List[str], int]]:
        counts = defaultdict(int)
        for txt_file in text_paths:
            with open(txt_file, "r") as f:
                texts = f.read()
                tokens = self.re_token.findall(texts)

                for token, count in Counter(tokens).items():
                    counts[token] += count
        # fixing \n to be <NL>
        # <NL> added thru special tokens
        # counts["<NL>"] += counts["\n"]
        del counts["\n"]

        bytes_count = []
        for token, count in counts.items():
            chars = [*token]
            for i, char in enumerate(chars[1:], 1):
                chars[i] = f"##{char}"
            bytes_count.append((chars, count))
        return bytes_count

    def count_byte_pairs(self, bytes_count):
        bp_counts = defaultdict(int)
        for byte, count in bytes_count:
            bigrams = [(l, r) for l, r in zip(byte[:-1], byte[1:])]
            bigram_counts = Counter(bigrams)
            for bigram, count in bigram_counts.items():
                bp_counts[bigram] += count
        return bp_counts

    def combine_byte_pair(self, byte_pair, bytes_count, joined):
        for i, (bytess, count) in enumerate(bytes_count):
            j = 0
            while j < len(bytess) - 1:
                if tuple(bytess[j : j + 2]) == byte_pair:
                    bytess = [*bytess[:j], joined, *bytess[j + 2 :]]
                else:
                    j += 1
            bytes_count[i] = (bytess, count)
        return bytes_count
