import os
import torch
from model import GPT, GPTConfig
from tokenizer import get_tokenizer
from config import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _normalize_rating(rating_str):
	try:
		r = float(rating_str)
	except Exception:
		return 0.5
	# simple heuristics: if likely 1-5 scale map to 0..1, else 0..10
	if 0 < r <= 5 and r.is_integer():
		return max(0.0, min(1.0, (r - 1) / 4.0))
	return max(0.0, min(1.0, r / 10.0))

def run_tune():
	os.system("clear")
	print("=== Interactive Tuning ===")
	tokenizer = get_tokenizer()
	vocab_size = tokenizer.get_vocab_size()

	cfg = GPTConfig(vocab_size)
	model = GPT(cfg).to(device)

	model_path = input("Path to model checkpoint (e.g. data/models/model.pt): ").strip()
	if not os.path.exists(model_path):
		print(f"Checkpoint '{model_path}' not found.")
		return

	# Load weights
	state = torch.load(model_path, map_location=device)
	model.load_state_dict(state)
	model.eval()

	prompt = input("Enter prompt: ").strip()
	if not prompt:
		print("Empty prompt, aborting.")
		return

	# Generate text (uses model.generate which is deterministic/greedy in model.py)
	try:
		generated = model.generate(prompt, max_new_tokens=MAX_GEN_TOKENS, tokenizer=tokenizer, device=device)
	except Exception:
		# Fallback generation
		input_ids = tokenizer.encode(prompt)
		x = torch.tensor([input_ids], dtype=torch.long).to(device)
		generated = ""
		with torch.no_grad():
			for _ in range(MAX_GEN_TOKENS):
				if x.size(1) >= model.pos_embed.num_embeddings:
					break
				logits, _ = model(x)
				logits = logits[:, -1, :] / TEMPERATURE
				probs = torch.softmax(logits, dim=-1)
				next_id = torch.multinomial(probs, num_samples=1)
				x = torch.cat([x, next_id], dim=1)
				try:
					token_str = tokenizer.decode([next_id.item()])
				except Exception:
					token_str = "�"
				if token_str == "~":
					break
				generated += token_str

	print("\n--- Generation ---")
	print(generated)
	print("------------------\n")

	rating_input = input("Rate this generation (0-10 or 1-5), higher = better: ").strip()
	rating_norm = _normalize_rating(rating_input)
	print(f"Normalized rating: {rating_norm:.3f}")

	# Build single training example from prompt + generated text
	full_text = prompt + generated
	token_ids = tokenizer.encode(full_text)
	if len(token_ids) < 2:
		print("Not enough tokens to train on.")
		return

	x_ids = token_ids[:-1]
	y_ids = token_ids[1:]
	x_tensor = torch.tensor([x_ids], dtype=torch.long).to(device)
	y_tensor = torch.tensor([y_ids], dtype=torch.long).to(device)

	model.train()
	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

	# Forward
	_, base_loss = model(x_tensor, y_tensor)
	if base_loss is None:
		print("Could not compute loss on this example.")
		return

	# Convert rating to factor in [-1,1]: positive -> reinforce (gradient ascent), negative -> penalize (descent)
	factor = (rating_norm - 0.5) * 2.0  # -1 .. 1
	loss_for_backward = -factor * base_loss
	if abs(factor) < 1e-6:
		loss_for_backward = base_loss

	optimizer.zero_grad()
	loss_for_backward.backward()
	if GRAD_CLIP is not None:
		torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
	optimizer.step()

	print(f"Performed one tuning step (rating_norm={rating_norm:.3f}, factor={factor:.3f}).")

	save_ans = input(f"Save updated model over '{model_path}'? (y/n): ").strip().lower()
	if save_ans == "y":
		os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
		torch.save(model.state_dict(), model_path)
		print(f"Saved updated model to {model_path}")
	else:
		print("Not saved.")

if __name__ == "__main__":
	run_tune()
