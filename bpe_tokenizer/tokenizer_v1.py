import time
from typing import List, Dict
from collections import defaultdict
from collections import Counter
from bpe_tokenizer_param import BPETokenizerParams
from bpe_tokenizer import BPETokenizer
import regex as re

def train_bpe(text: str, num_merges: int) -> BPETokenizerParams:
    print(f"Training BPE on: '{text}' with {num_merges} merges...")

    indicies = list(text.encode("utf-8"))
    print("Indicies: ", indicies)
    merges: dict[tuple[int, int], int] = {}
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}

    for i in range(num_merges):
        counts = count_pair(indicies)
        if len(counts) == 0:
            print("No more pairs.")
            break

        print("Counts: ", counts)

        # Find the most common pair
        max_pair = select_max_pair(counts)

        new_index = generate_new_index(i)

        merges[max_pair] = new_index

        update_vocab(vocab, max_pair, new_index)

        indicies = merge(indicies, max_pair, new_index)

    return BPETokenizerParams(vocab, merges)

def generate_new_index(i):
    new_index = 256 + i
    print("New index: ", new_index)
    return new_index

def update_vocab(vocab, pair, new_index):
    index1, index2 = pair
    print("Vocab before update: ", vocab)
    vocab[new_index] = vocab[index1] + vocab[index2]
    print("Updated vocab: ", vocab)

def select_max_pair(counts):
    max_pair = max(counts, key=counts.get)
    print("Max pair: ", max_pair)
    return max_pair

def count_pair(indicies):
    counts = defaultdict(int)

    for index1, index2 in zip(indicies, indicies[1:]):
        counts[(index1, index2)] += 1

    return counts

def merge(indicies: list[int], pair: tuple[int, int], new_index: int) -> list[int]:

    new_indicies = []

    i = 0
    while i < len(indicies):
        if i+1 < len(indicies) and indicies[i] == pair[0] and indicies[i+1] == pair[1]:
            new_indicies.append(new_index)
            i += 2
        else:
            new_indicies.append(indicies[i])
            i += 1

    print("New indicies: ", new_indicies)
    return new_indicies


if __name__ == "__main__":
    start_time = time.time()
    
    params = train_bpe("the cat in the hat", 1)

    tokenizer = BPETokenizer(params)
    string = "the quick brown fox"
    print("String for encoding:", string)
    indices = tokenizer.encode(string)
    reconstructed_string = tokenizer.decode(indices)
    print("String after decoding:", string)
    assert string == reconstructed_string

    print("Execution time:", round(time.time() - start_time, 3), "seconds")

