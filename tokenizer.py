from config import *
import os
import re
import unicodedata

os.system("clear")
def get_auto_vocab_size(cleaned_dir):
    unique_chars = set()
    for f in os.listdir(cleaned_dir):
        if f.endswith(".txt"):
            with open(os.path.join(cleaned_dir, f), "r", encoding="utf-8") as fin:
                for line in fin:
                    unique_chars.update(line.strip())
    # At minimum, return 1 to avoid errors
    return max(1, len(unique_chars))

class CharTokenizer:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.stoi = {ch: i for i, ch in enumerate(alphabet)}
        self.itos = {i: ch for i, ch in enumerate(alphabet)}
    def encode(self, text):
        return [self.stoi[ch] for ch in text if ch in self.stoi]
    def decode(self, ids):
        return ''.join([self.itos[i] for i in ids if i in self.itos])
    def get_vocab_size(self):
        return len(self.alphabet)

def get_tokenizer():
    cleaned_dir = "data/training_data"
    if not os.path.exists(cleaned_dir):
        os.makedirs(cleaned_dir, exist_ok=True)
        # Clean all .txt files in DATA_PATH
        for f in os.listdir(DATA_PATH):
            if f.endswith(".txt"):
                clean_text_file(os.path.join(DATA_PATH, f), os.path.join(cleaned_dir, f))

    # Always use character-level tokenizer
    return CharTokenizer(TOKENIZER_ALPHABET)
    retrain = False
    if vocab_size == 0:
        vocab_size = get_auto_vocab_size(cleaned_dir)
        print(f"[Auto] Setting TOKENIZER_VOCAB_SIZE to {vocab_size} (unique tokens in cleaned data)")
        # Always retrain in auto mode
        retrain = True

    # Remove old tokenizer files if retraining
    if retrain:
        if os.path.exists("tokenizer/vocab.json"):
            os.remove("tokenizer/vocab.json")
        if os.path.exists("tokenizer/merges.txt"):
            os.remove("tokenizer/merges.txt")

    if not os.path.exists("tokenizer/vocab.json"):
        tokenizer = ByteLevelBPETokenizer()

        # Use cleaned files for tokenizer training
        files = [
            os.path.join(cleaned_dir, f)
            for f in os.listdir(cleaned_dir)
            if f.endswith(".txt")
        ]

        if not files:
            raise FileNotFoundError("No .txt files found in the cleaned training data.")

        # Use TOKENIZER_ALPHABET from config
        tokenizer.train(
            files=files,
            vocab_size=vocab_size,
            min_frequency=TOKENIZER_MIN_FREQUENCY,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
        )
        os.makedirs("tokenizer", exist_ok=True)  # Ensure directory exists
        tokenizer.save_model("tokenizer")  # This saves as vocab.json and merges.txt
        print("Tokenizer saved to 'tokenizer/'")
        print("Vocab size:", tokenizer.get_vocab_size())
        print("Sample vocab:", list(tokenizer.get_vocab().keys())[:20])
    else:
        tokenizer = ByteLevelBPETokenizer(
            "tokenizer/vocab.json",
            "tokenizer/merges.txt"
        )
    # Fix: Avoid recursion by referencing the original method
    orig_get_vocab_size = tokenizer.get_vocab_size
    def _get_vocab_size():
        return orig_get_vocab_size()
    tokenizer.get_vocab_size = _get_vocab_size
    return CharTokenizer(TOKENIZER_ALPHABET)
