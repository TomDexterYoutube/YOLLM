import torch
from config import *


# =========================================================
# Repetition penalty
# Applied to ALL tokens seen so far (prompt + generated).
# Divides positive logits, multiplies negative logits so the
# effect is always a reduction in probability.
# =========================================================

def apply_repetition_penalty(logits, prompt_ids, generated_ids, penalty):
    if penalty == 1.0:
        return logits

    seen = list(set(prompt_ids) | set(generated_ids))
    if not seen:
        return logits

    # Vectorised: gather the logits for all seen tokens, apply penalty, scatter back
    seen_t  = torch.tensor(seen, dtype=torch.long, device=logits.device)
    scores  = logits[0, seen_t]
    scores  = torch.where(scores > 0, scores / penalty, scores * penalty)
    logits[0, seen_t] = scores

    return logits


# =========================================================
# N-gram repetition block
# Prevents the model from repeating any n-gram that already
# appeared in prompt + generated context.
# =========================================================

def apply_ngram_block(logits, prompt_ids, generated_ids, n):
    if n <= 0 or len(generated_ids) < n - 1:
        return logits

    context  = generated_ids[-(n - 1):]
    all_ids  = prompt_ids + generated_ids

    for i in range(len(all_ids) - n + 1):
        if all_ids[i : i + n - 1] == context:
            blocked = all_ids[i + n - 1]
            logits[0, blocked] = float("-inf")

    return logits


# =========================================================
# Nucleus (top-p) sampling
# =========================================================

def sample(logits, temperature=TEMPERATURE, top_p=TOP_P):
    logits = logits / max(temperature, 1e-8)
    probs  = torch.softmax(logits, dim=-1)

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Shift cumulative right by one: the token that *pushes* cumulative
    # over top_p is the last one to keep, not the first to drop.
    # Without the shift, we always drop at least one valid token, making
    # the distribution too narrow (especially at low top_p values).
    remove_mask = cumulative - sorted_probs > top_p
    sorted_probs[remove_mask] = 0.0

    # Safety: ensure at least one token survives
    if sorted_probs.sum() == 0:
        sorted_probs[0] = 1.0

    sorted_probs = sorted_probs / sorted_probs.sum()

    next_idx_in_sorted = torch.multinomial(sorted_probs, 1).item()
    return sorted_idx[0, next_idx_in_sorted].item()


# =========================================================
# Main generation function
# =========================================================

def generate_response(prompt, model, tokenizer, device, autocast_fn,
                      max_new_tokens=MAX_GEN_TOKENS, stream=False):

    model.eval()

    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        return ""
    x          = torch.tensor([prompt_ids], device=device)
    generated  = []

    # Use model.block_size — always present; never rely on pos_embed
    max_len = model.block_size

    with torch.inference_mode(), autocast_fn():
        for _ in range(max_new_tokens):

            if x.shape[1] >= max_len:
                break

            logits, _ = model(x)
            logits    = logits[:, -1, :].float().clone()  # clone: in-place ops below need a mutable tensor

            logits = apply_repetition_penalty(
                logits, prompt_ids, generated, REPETITION_PENALTY
            )
            logits = apply_ngram_block(
                logits, prompt_ids, generated, NO_REPEAT_NGRAM_SIZE
            )

            next_id = sample(logits)

            x         = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
            generated.append(next_id)

            piece = tokenizer.decode([next_id])

            # Stop on EOS marker
            if piece == "~":
                break

            if stream:
                print(piece, end="", flush=True)

    if stream:
        print()

    # Do NOT call model.train() here — the caller owns the model mode.
    # Calling it here was silently switching the model to train mode
    # inside write_progress / write_loss_resp which already handle mode.
    return tokenizer.decode(generated)