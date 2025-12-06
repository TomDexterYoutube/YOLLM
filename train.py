import os
import time
import torch
import gc
import psutil
import signal
import sys
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import GPTConfig, GPT
from tokenizer import get_tokenizer
from config import *
import threading
from lion_pytorch import Lion

# Overwrite the file to clear it at the start of training
with open("epoch_generations.txt", "w") as f:
    f.write("")
    pass

latest_cpu_temp = [None]
can_train = threading.Event()
can_train.set()  # Allow training by default

# New: Lock to pause/resume training instantly
train_lock = threading.Lock()
train_lock.acquire()  # Start locked, will be released if temp is OK

def cpu_temp_monitor(poll_interval=1):
    while True:
        temp = None
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                temp = int(f.read()) / 1000.0
        except Exception:
            pass
        latest_cpu_temp[0] = temp

        # Control training permission
        if temp is not None and temp >= 87:
            if can_train.is_set():
                print(f"CPU temperature is {temp:.1f}°C. Pausing training for safety.")
            can_train.clear()
            if train_lock.locked() is False:
                train_lock.acquire()
        elif temp is not None and temp <= 60:
            if not can_train.is_set():
                print(f"CPU temperature is {temp:.1f}°C. Resuming training.")
            can_train.set()
            if train_lock.locked():
                try:
                    train_lock.release()
                except RuntimeError:
                    pass

        time.sleep(poll_interval)

# Start the background temperature monitor thread
temp_thread = threading.Thread(target=cpu_temp_monitor, args=(5,), daemon=True)
temp_thread.start()

def get_cpu_temp():
    return latest_cpu_temp[0]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

# Load tokenizer
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()

# Dataset class
class TextDataset(Dataset):
    def __init__(self, folder_path, block_size):
        data = ""
        # Recursively walk through all subfolders and files
        for root, _, files in os.walk(folder_path):
            for file_name in files:
                if file_name.endswith(".txt"):
                    file_path = os.path.join(root, file_name)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data += f.read() + "\n"
        # CharTokenizer.encode returns a list, not an object with .ids
        self.tokens = tokenizer.encode(data)
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.tokens) - self.block_size)

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

# Setup data
dataset = TextDataset("data/training_data/", BLOCK_SIZE)
dataset_size = len(dataset)

if dataset_size == 0:
    print("ERROR: Dataset is empty.")
    exit()

# Define variables before using them
block_size = BLOCK_SIZE  # From config.py
batch_size = BATCH_SIZE  # From config.py

# Setup data with weighted sampling
def get_sample_weights(dataset):
    # Weight rare tokens more heavily
    token_counts = torch.bincount(torch.tensor(dataset.tokens))
    token_weights = 1.0 / (token_counts + 1)  # Add 1 to avoid division by zero
    sample_weights = torch.zeros(len(dataset))
    for i in range(len(dataset)):
        chunk = dataset.tokens[i:i + dataset.block_size]
        sample_weights[i] = token_weights[chunk].mean()
    return sample_weights

# Create dataset before using it
dataset = TextDataset("data/training_data/", block_size)

# Create weighted sampler
sample_weights = get_sample_weights(dataset)
sampler = torch.utils.data.WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset),
    replacement=True
)

dataloader = DataLoader(
    dataset, 
    batch_size=batch_size,
    sampler=sampler if SHUFFLE_DATA_EACH_EPOCH else None,
    num_workers=2,
    pin_memory=True if device == 'cuda' else False
)

# Build model
model = GPT(GPTConfig(vocab_size=vocab_size, block_size=block_size)).to(device)

optimizer = Lion(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=(0.7, 0.8),
)

# Optional resume from checkpoint
checkpoint_path = "data/models/model.pt"
epoch_path = "data/models/last_epoch.txt"
resume = False
start_epoch = 0
if os.path.exists(checkpoint_path):
    ans = input(f"Checkpoint found at {checkpoint_path}. Resume training from checkpoint? (y/n): ").strip().lower()
    if ans == "y":
        resume = True

if resume:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Resumed model weights from {checkpoint_path}")
    if os.path.exists(epoch_path):
        try:
            with open(epoch_path, "r") as f:
                start_epoch = int(f.read().strip())
            print(f"Resuming from epoch {start_epoch + 1}")
        except Exception:
            start_epoch = 0
    else:
        start_epoch = 0

# ACT Tracking for all epochs
total_batches = EPOCHS * len(dataloader)
start_time = time.time()
batches_done = 0

def generate_text(model, prompt, tokenizer, device, max_new_tokens=MAX_TRAIN_TOKENS):
    config = ModelConfig()  # Get generation parameters
    model.eval()
    input_ids = tokenizer.encode(prompt)
    if hasattr(input_ids, "ids"):
        input_ids = input_ids.ids
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(input_ids)
            next_token_logits = logits[:, -1, :] / config.temperature
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                for i in range(input_ids.shape[1]):
                    next_token_logits[0, input_ids[0, i]] /= config.repetition_penalty
            
            # Apply top-p (nucleus) sampling
            if config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > config.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check minimum length
            if input_ids.shape[1] < config.min_length:
                continue
                
            # Apply no-repeat-ngram
            if config.no_repeat_ngram_size > 0 and input_ids.shape[1] >= config.no_repeat_ngram_size:
                ngram = input_ids[0, -config.no_repeat_ngram_size:].tolist()
                if ngram in input_ids[0, :-config.no_repeat_ngram_size].tolist():
                    break
    try:
        return tokenizer.decode(input_ids[0].tolist())
    except Exception as e:
        return f"[DECODE ERROR: {e}]"

def cleanup():
    torch.cuda.empty_cache()
    gc.collect()

def sync_filesystem():
    os.system('sync')  # Force write buffers to disk

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_percent()

# Graceful shutdown handler
def signal_handler(signum, frame):
    print("\nReceived shutdown signal. Cleaning up...")
    cleanup()
    sync_filesystem()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def calculate_penalties(logits, targets, generated_text):
    # Similarity penalty
    probs = F.softmax(logits, dim=-1)
    max_prob_penalty = torch.max(probs, dim=-1)[0].mean() * SIMILARITY_PENALTY
    
    # Diversity calculations
    unique_tokens = len(torch.unique(targets))
    total_tokens = targets.numel()
    diversity_score = unique_tokens / total_tokens
    diversity_penalty = (1 - diversity_score) * DIVERSITY_FACTOR
    
    # Complexity reward
    complexity_bonus = -COMPLEXITY_REWARD if unique_tokens >= MIN_UNIQUE_TOKENS else 0
    
    return max_prob_penalty + diversity_penalty + complexity_bonus

def analyze_generation_quality(text, tokenizer):
    """Analyze generated text quality and return a loss adjustment factor."""
    if not text or len(text) < 5:
        return 0.5  # Low quality - very short text
    
    # Penalize repetitive patterns
    words = text.split()
    unique_ratio = len(set(words)) / max(len(words), 1)
    repetition_penalty = 1.0 - (unique_ratio * 0.3)
    
    # Reward coherent length
    length_bonus = min(len(text) / 200, 1.0) * 0.2
    
    # Penalize common placeholder text
    low_quality_markers = ["in in in", "sss", "ish", "~~~", "..."]
    marker_penalty = 0.0
    for marker in low_quality_markers:
        if marker in text.lower():
            marker_penalty += 0.3
    
    # Reward diversity in character usage
    unique_chars = len(set(text))
    diversity_bonus = min(unique_chars / 50, 1.0) * 0.2
    
    # === NEW: Check for reward words ===
    reward_word_bonus = 0.0
    text_lower = text.lower()
    for word in REWARD_WORDS:
        if word in text_lower:
            reward_word_bonus += WORD_REWARD_BONUS
    
    # === NEW: Check for penalize words ===
    penalize_word_penalty = 0.0
    for word in PENALIZE_WORDS:
        if word in text_lower:
            penalize_word_penalty += WORD_PENALTY_BONUS
    
    quality_score = (
        repetition_penalty + 
        length_bonus + 
        diversity_bonus - 
        marker_penalty +
        reward_word_bonus -
        penalize_word_penalty
    )
    
    return max(0.1, min(quality_score, 1.5))  # Clamp between 0.1 and 1.5

for epoch in range(start_epoch, EPOCHS):
    # Wait for lock to be released before starting each epoch
    train_lock.acquire()
    train_lock.release()
    model.train()
    loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
    total_loss = 0

    for x, y in loop:
        # Check memory usage and cleanup if needed
        if get_memory_usage() > 90:  # If using >90% of system memory
            print("\nHigh memory usage detected. Running cleanup...")
            cleanup()
        
        train_lock.acquire()
        train_lock.release()
        x, y = x.to(device), y.to(device)
        
        try:
            logits, base_loss = model(x, y)
            
            # Calculate combined penalties
            penalties = calculate_penalties(logits, y, None)
            
            # Final loss with all penalties
            loss = base_loss + penalties
            
            optimizer.zero_grad()
            loss.backward()
            if GRAD_CLIP is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()
        except RuntimeError as e:
            print(f"\nError during training: {e}")
            cleanup()
            continue

        # Periodic cleanup every 10 batches
        if batches_done % 10 == 0:
            cleanup()
            sync_filesystem()  # Sync to prevent filesystem corruption

        # ACT for all training
        batches_done += 1
        elapsed = time.time() - start_time
        time_per_batch = elapsed / batches_done
        remaining = (total_batches - batches_done) * time_per_batch
        mins, secs = divmod(remaining, 60)
        act_str = f"{int(mins):02d}:{int(secs):02d}"

        loop.set_postfix(loss=loss.item(), ACT=act_str)

    # After each epoch
    cleanup()
    sync_filesystem()
    # Save after each epoch
    os.makedirs(f"data/models/epoch{(epoch)+1}", exist_ok=True)
    torch.save(model.state_dict(), f"data/models/epoch{(epoch)+1}/model.pt")
    torch.save(model.state_dict(), f"data/models/model.pt")
    # Save last completed epoch
    with open(epoch_path, "w") as f:
        f.write(str(epoch + 1))

    with open("epoch_generations.txt", "a") as f:
        test_prompts = ["hi", "hello", "how are you", "what is", "tell me about"]

        for prompt in test_prompts:
            generated = generate_text(prompt="user: " + prompt + "~", model=model, tokenizer=tokenizer, device=device, max_new_tokens=MAX_TRAIN_TOKENS)
            
            # === NEW: Detect words in generation ===
            found_reward = [w for w in REWARD_WORDS if w in generated.lower()]

            f.write(f"Prompt: '{prompt}'\nGeneration: {generated}\n")
            if found_reward:
                loss = loss - WORD_REWARD
            else:
                loss = loss + WORD_PENALTY

# Final cleanup
cleanup()
sync_filesystem()
print("Training complete.")
print()
print("epoch auto test data:")
with open("epoch_generations.txt", "r") as a:
    print(a.read())
input("Press Enter to exit...")