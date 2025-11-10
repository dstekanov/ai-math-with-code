import time
import regex as re
from collections import Counter
from typing import Dict, List, Tuple
from bpe_tokenizer_param import BPETokenizerParams

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(text: str, num_merges: int, special_tokens: list[str]) -> BPETokenizerParams:
    print(f"Training BPE on: '{text}' with {num_merges} merges...")

    pre_counts = pretokenize(text, special_tokens)
    token_indicies, freqs = tokens_to_indicies_lists(pre_counts)
    vocab = build_initial_vocab()
    next_id = update_vocab_with_special_tokens(vocab, special_tokens)

    merges: Dict[Tuple[int, int], int] = {}

    for i in range(num_merges):
        pair_counts = count_pair_frequencies(token_indicies, freqs)

        if not pair_counts:
            print("No more pairs to merge.")
            break

        best_pair, best_count = max(pair_counts.items(), key=lambda item: (item[1], item[0]))       

        if best_count < 2:
            print("No pairs with count >= 2. Stopping.")
            break

        new_id = next_id
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges[best_pair] = new_id
        next_id += 1

        print(f"Merge #{i+1}: {best_pair} w/ count {best_count} -> {new_id}, bytes={vocab[new_id]!r}")

        token_indicies = [
            apply_merge_on_token(s, best_pair, new_id) for s in token_indicies
        ]

    return BPETokenizerParams(vocab, merges)

def update_vocab_with_special_tokens(vocab, special_tokens: list[str]):
    next_id = max(vocab.keys()) + 1

    for special_token in special_tokens:
        vocab[next_id] = special_token.encode("utf-8")
        next_id += 1
    
    return next_id

def pretokenize(text: str, special_tokens: list[str]) -> Counter:
    """Return Counter of pre-tokens using GPT-2 regex pattern."""

    chunks = re.split("|".join(re.escape(token) for token in special_tokens), text)

    matches = []
    for chunk in chunks:
        matches.extend(re.finditer(PAT, chunk))
    
    return Counter(m.group() for m in matches)

def build_initial_vocab() -> Dict[int, bytes]:
    """Create the initial byte-level vocabulary: id -> single-byte bytes."""
    return {i: bytes([i]) for i in range(256)}

def tokens_to_indicies_lists(pre_counts: Counter):
    token_indicies = []
    freqs = []
    for token, freq in pre_counts.items():
        b = token.encode("utf-8")
        token_indicies.append(list(b))
        freqs.append(freq)
    return token_indicies, freqs

def count_pair_frequencies(token_indicies: List[List[int]], freqs: List[int]) -> Counter:
    """Count how often each adjacent byte-pair occurs, weighted by token frequency."""
    counts = Counter()
    for indicies, f in zip(token_indicies, freqs):
        for i in range(len(indicies) - 1):
            counts[(indicies[i], indicies[i + 1])] += f
    return counts

def apply_merge_on_token(indicies: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """Replace non-overlapping occurrences of pair (a,b) with new_id."""
    a, b = pair
    new_indicies = []
    i = 0
    while i < len(indicies):
        if i < len(indicies) - 1 and indicies[i] == a and indicies[i + 1] == b:
            new_indicies.append(new_id)
            i += 2
        else:
            new_indicies.append(indicies[i])
            i += 1
    return new_indicies


if __name__ == "__main__":
    start_time = time.time()

    special_tokens = ["<|endoftext|>"]

    text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest low low low low low " \
    "lower lower widest widest widest <|endoftext|> newest newest newest newest newest newest " \
    "low low low low low lower lower widest widest widest newest newest newest newest newest newest"

    result = train_bpe(text, 15, special_tokens)

    print("Execution time:", round(time.time() - start_time, 3), "seconds")

