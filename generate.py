import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from model import GPT, GPTConfig
from config import *
from tokenizer import get_tokenizer


device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()

config = GPTConfig(vocab_size)
model = GPT(config).to(device)

print(f"Tokenizer vocab size: {vocab_size}")
print(f"Model embedding size: {model.token_embed.num_embeddings}")
if vocab_size != model.token_embed.num_embeddings:
    print("[ERROR] Tokenizer vocab size does not match model embedding size!")
    print("You must retrain your model with the current tokenizer.")

os.system("clear")
os.system("dir data/models")
model_path = "data/models/" + input("model_path/")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model checkpoint '{model_path}' not found. Train the model first.")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

EOS_TOKEN = "~"  # Make sure this is in your TOKENIZER_ALPHABET

def generate_text(prompt, max_new_tokens=MAX_GEN_TOKENS):
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long).to(device)
    generated = ""
    for _ in range(max_new_tokens):
        if x.shape[1] >= model.pos_embed.num_embeddings:
            print("[WARNING] Generation stopped: reached model block size limit.")
            break
        with torch.no_grad():
            logits, _ = model(x)
        logits = logits[:, -1, :]
        logits = logits / TEMPERATURE
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)
        try:
            next_token_str = tokenizer.decode([next_id.item()])
        except Exception as e:
            next_token_str = f"[DECODE ERROR: {e}]"
        # Show each token as it is generated
        print(next_token_str, end="", flush=True)
        if next_token_str == EOS_TOKEN:
            break
        generated += next_token_str
    print()  # Newline after generation
    return generated

if __name__ == "__main__":
    # Utility: Check tokenizer decode coverage
    print("Checking tokenizer decode coverage...")
    undecodable = 0
    for i in range(tokenizer.get_vocab_size()):
        try:
            s = tokenizer.decode([i])
            if "�" in s:
                print(f"Token {i} decodes to undecodable char: {s}")
                undecodable += 1
        except Exception as e:
            print(f"Token {i} decode error: {e}")
            undecodable += 1
    print(f"Total undecodable tokens: {undecodable} / {tokenizer.get_vocab_size()}")

    while True:
        prompt = "user: " + input("user: ") + "~"
        if prompt.lower() in ["exit()"]:
            break
        generate_text(prompt)
