# BPE Tokenizer - My First Naive Implementation

## Finding What Really Drives Me in AI

It all started with a search for "what truly interests me in AI." I wanted to find that part that would drive me, where I'd be genuinely curious.

Recently, I stopped at an interesting course from Stanford - CS336. They freely posted their video lectures on YouTube, and they have a website with all assignments and code.

I actually saw their course when it first came out, and now I stopped, realizing this is what I want.

After watching the first lecture, I approached the first assignment. The goal - training a tokenizer.

## The Stanford Mindset

From the very beginning, I wanted to adopt the mindset of Stanford students:

**Mindset of Stanford CS336 Students:**
- They don't memorize algorithms — they reconstruct them from principles
- They debug by visualization — print out merge tables, vocabulary changes, token sequences
- They collaborate — discuss high-level ideas, but each writes their own code
- They reflect — every assignment ends with "what surprised me" or "what I'd improve"

## The Journey Through Unknown Territory

### What is a Tokenizer and Why Do We Need It?

Looking at the minimal implementation:

**Input:**
- `string` - my sentence
- `num_merges` - how many times to merge? Why this? Why should I choose how many times to merge?

**Process:**

1. **Indices** - list of bytes. We encode the string to UTF-8.
   - Remember: what is encoding and what types exist?
   - What's the difference? What does different encoding affect?
   - What are alternatives to get bytes from a string?

2. **Merges** - result of merged indices

3. **Vocab** - mapping of index to bytes... didn't understand at first

### The Training Loop

Loop through `num_merges`:

**Count adjacent pairs:**
- Why only adjacent?

**Find the most frequent pair:**
- Why only one in the current cycle?

**Generate new index for the pair:**
- `256 + i`
- Is 256 the maximum UTF-8 number?

**Add to merges:**
- pair + new index

**Update vocab:**
- new index = sum of previous bytes
- (still don't understand why we need it)

**Merge indices:**
- Loop through indices
- First condition: if we found a pair and checked we won't go out of array bounds
  - Add to new indices array (new index and skip these two indices i+2)
- Otherwise, also add current index, forming a new list

Can this be improved?

**Return:**
- Trained vocabulary and merges (which pairs were merged) in the form of `BPETokenizerParams` class

## The Tokenizer Classes

### BPETokenizer Class

Takes our `params` class and has two methods: encode and decode

**Encode:**
- Converts string to bytes
- Creates a for loop through trained merges
- Calls our existing merge method
- Returns indices

**Decode:**
- Creates a list of bytes from vocab by indices (need to debug this)
- Returns string joined and decoded by UTF-8

## Implementation Files

- `tokenizer_v1.py` - Main training implementation with `train_bpe()` function
- `bpe_tokenizer.py` - BPETokenizer class with encode/decode methods
- `bpe_tokenizer_param.py` - Data class for storing vocab and merges
- `test_bpe.py` - Tests for the training process
- `test_bpe_tokenizer.py` - Tests for the tokenizer class

## What I Learned

This is my first naive version of the BPE train method. It's not optimized, but it helped me understand:
- How tokenizers work from first principles
- Why we need byte-level encoding
- The merge algorithm and vocabulary building
- The relationship between training and inference

## Next Steps

- Optimize the merge operation (currently O(n) per merge)
- Understand vocab usage better
- Add more comprehensive tests
- Study production implementations (tiktoken, sentencepiece)
