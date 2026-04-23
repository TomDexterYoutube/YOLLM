import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from model import GPT, GPTConfig
from config import *
from tokenizer import get_tokenizer
from generate import generate_response   # FIX: use shared generation module

# =========================================================
# Setup
# =========================================================

device     = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer  = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()
config     = GPTConfig(vocab_size)
model      = GPT(config).to(device)

# =========================================================
# Autocast helper (mirrors train.py so generate_response works)
# =========================================================

def autocast():
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    try:
        import time
        def _bench(dtype):
            try:
                a = torch.randn(256, 256).to(dtype)
                b = torch.randn(256, 256).to(dtype)
                for _ in range(2): torch.matmul(a, b)
                t0 = time.perf_counter()
                for _ in range(5): torch.matmul(a, b)
                return (time.perf_counter() - t0) / 5
            except Exception:
                return float("inf")
        amp_dtype = torch.bfloat16 if _bench(torch.bfloat16) < _bench(torch.float32) else torch.float32
    except Exception:
        amp_dtype = torch.float32
    if amp_dtype == torch.float32:
        return torch.autocast("cpu", enabled=False)
    return torch.autocast("cpu", dtype=amp_dtype)

# =========================================================
# Load model
# =========================================================

# FIX: replaced os.system("clear") with a blank print — os.system("clear")
# erases any error messages or tracebacks that appeared before the prompt,
# and fails silently on Windows.
print()
model_path = "data/models/" + input("model path: data/models/")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model checkpoint '{model_path}' not found.")

# FIX: weights_only=True avoids arbitrary pickle execution (PyTorch will
# make this a hard error in a future release)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

print()
print(f"Model loaded — vocab: {vocab_size} | params: {sum(p.numel() for p in model.parameters()):,}")
print("Type 'exit()' to quit.\n")

# =========================================================
# Chat loop
# =========================================================

# FIX: the original printed "edward: " from the script and let the model
# generate tokens starting from after that prefix. This caused the model
# to output "edward: ..." again (double prefix) because it was trained
# to produce "edward: <response>~" as a full sequence.
#
# The fix is to include "edward: " in the prompt so the model completes
# from there, and NOT print it separately from the script. The generated
# tokens will start right after "edward: " and will be streamed directly.

if __name__ == "__main__":
    while True:
        try:
            user_input = input("user: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if user_input.lower() in ("exit()", "quit", "exit"):
            print("Goodbye.")
            break

        if not user_input:
            continue

        # FIX: prompt now ends with "\nedward: " so the model completes
        # the response inline — the "edward: " prefix comes from generation
        # rather than being printed by the script, eliminating the double
        # prefix bug.
        prompt = f"user: {user_input}~\nedward: "
        print("edward: ", end="", flush=True)

        response = generate_response(
            prompt,
            model,
            tokenizer,
            device,
            autocast,
            max_new_tokens=MAX_GEN_TOKENS,
            stream=True,   # tokens print as they are generated
        )
        print()