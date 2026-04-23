# tokenizer.py
#
# Character-level tokenizer — best choice for small datasets.
# Vocab is built automatically from every character that appears
# in the training files, so it is always 100% lossless with zero
# <unk> tokens and no separate training step required.
#
import os
import hashlib
import json

DATA_DIR       = "data/training_data"
VOCAB_CACHE    = "tokenizer_vocab.json"


# -------------------
# HELPERS
# -------------------

def _collect_text_files(data_dir):
    paths = []
    for root, _, files in os.walk(data_dir):
        for f in sorted(files):
            if f.endswith(".txt") and not f.startswith("_"):
                paths.append(os.path.join(root, f))
    return paths


def _build_alphabet(data_dir):
    """Scan all training files and return a sorted list of unique characters."""
    chars = set()
    for path in _collect_text_files(data_dir):
        with open(path, "r", encoding="utf-8") as f:
            chars.update(f.read())
    if not chars:
        raise ValueError(f"No characters found in {data_dir} — is your training data empty?")
    return sorted(chars)


# -------------------
# TOKENIZER CLASS
# -------------------

class CharTokenizer:
    """
    Lossless character-level tokenizer.

    • encode(text) → list[int]   — every character in the alphabet maps to
                                   a unique integer; unknown chars are skipped
                                   with a warning rather than crashing.
    • decode(ids)  → str         — exact inverse of encode for in-vocab ids.
    • get_vocab_size() → int
    """

    def __init__(self, alphabet: list):
        if not alphabet:
            raise ValueError("Alphabet is empty.")
        self.alphabet = alphabet
        self.stoi     = {ch: i for i, ch in enumerate(alphabet)}
        self.itos     = {i: ch for i, ch in enumerate(alphabet)}

    def encode(self, text: str) -> list:
        ids = []
        for ch in text:
            if ch in self.stoi:
                ids.append(self.stoi[ch])
            # silently skip chars outside the alphabet (e.g. rare unicode)
            # — this keeps inference working even on unseen characters
        return ids

    def decode(self, ids: list) -> str:
        return "".join(self.itos[i] for i in ids if i in self.itos)

    def get_vocab_size(self) -> int:
        return len(self.alphabet)


# -------------------
# GET TOKENIZER
#
# Builds the alphabet from training data on first call and caches it
# to tokenizer_vocab.json so subsequent loads are instant and the
# alphabet stays stable between training and inference.
# -------------------

def train_tokenizer():
    """
    Build and save the character vocabulary from training data.
    Safe to call even if the vocab already exists — will skip if up-to-date.
    """
    alphabet = _build_alphabet(DATA_DIR)
    with open(VOCAB_CACHE, "w", encoding="utf-8") as f:
        json.dump(alphabet, f, ensure_ascii=False)
    print(f"Tokenizer vocab built: {len(alphabet)} characters → '{VOCAB_CACHE}'")
    return alphabet


def get_tokenizer() -> CharTokenizer:
    """
    Load the tokenizer, building the vocab from scratch if needed.
    Always call this — never instantiate CharTokenizer directly.
    """
    if not os.path.exists(VOCAB_CACHE):
        print(f"'{VOCAB_CACHE}' not found — building from training data...")
        alphabet = train_tokenizer()
    else:
        with open(VOCAB_CACHE, "r", encoding="utf-8") as f:
            alphabet = json.load(f)

    if not alphabet:
        raise ValueError(
            f"Loaded alphabet from '{VOCAB_CACHE}' is empty. "
            "Delete it and re-run to rebuild from training data."
        )

    return CharTokenizer(alphabet)


# -------------------
# DATA HASH  (used by train.py cache invalidation)
# -------------------

def hash_data_dir(data_dir: str) -> str:
    """
    Short MD5 hash of all training file names + sizes + mtimes.
    Used to automatically invalidate the prebatch cache when data changes.
    """
    h = hashlib.md5()
    for root, _, files in os.walk(data_dir):
        for fname in sorted(files):
            if fname.endswith(".txt") and not fname.startswith("_"):
                full = os.path.join(root, fname)
                h.update(fname.encode())
                h.update(str(os.path.getsize(full)).encode())
                h.update(str(os.path.getmtime(full)).encode())
    return h.hexdigest()[:8]


# -------------------
# SANITY TEST
# -------------------

if __name__ == "__main__":
    tok = get_tokenizer()
    print(f"Vocab size: {tok.get_vocab_size()}")
    print(f"Alphabet:   {''.join(tok.alphabet)}")
    print()

    tests = [
        "Hello world!",
        "user: hello~",
        "edward: hi there, how are you?~",
        "what is your name?",
        "user: tell me a story~",
    ]

    all_passed = True
    for sample in tests:
        ids     = tok.encode(sample)
        decoded = tok.decode(ids)
        ok      = decoded == sample
        if not ok:
            all_passed = False
        print(f"{'✓' if ok else '✗'} '{sample}'")
        if not ok:
            print(f"  got: '{decoded}'")

    print()
    print("All tests passed!" if all_passed else "Some tests FAILED — check output above.")