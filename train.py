# =========================================================
# Environment vars MUST be set before importing torch
# =========================================================
import os
os.environ["OMP_SCHEDULE"]        = "static"
os.environ["OMP_PROC_BIND"]       = "close"
os.environ["GOMP_SPINCOUNT"]      = "100000000"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["KMP_AFFINITY"]        = "granularity=fine,compact,1,0"
os.environ["KMP_BLOCKTIME"]       = "1"

import csv
import time
import torch
import gc
import signal
import sys
import logging
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from model import GPTConfig, GPT
from tokenizer import get_tokenizer, train_tokenizer, hash_data_dir
from generate import generate_response
from config import *

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# =========================================================
# Logging Setup
# =========================================================

os.makedirs("data/models", exist_ok=True)

log = logging.getLogger("train")
log.setLevel(logging.DEBUG)

fh = logging.FileHandler("training.log", mode="a", encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
))

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(message)s"))

log.addHandler(fh)
log.addHandler(ch)

def info(msg):   log.info(msg)
def detail(msg): log.debug(msg)
def warn(msg):   log.warning(msg)
def error(msg):  log.error(msg)

PROGRESS_LOG  = "progress.log"
LOSS_RESP_LOG = "loss-resp.log"

EVAL_PROMPTS = [
    "user: hi~",
    "user: how are you?~",
    "user: what is your name?~",
]

info("=" * 60)
info("Training started")
info("=" * 60)

# =========================================================
# Tokenizer
# =========================================================

# CharTokenizer builds its vocab from training data automatically.
# train_tokenizer() is a no-op if tokenizer_vocab.json already exists.
if not os.path.exists("tokenizer_vocab.json"):
    info("Tokenizer vocab not found — building from training data...")
    train_tokenizer()

tokenizer  = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()
detail(f"Tokenizer loaded | vocab size: {vocab_size} characters")

# =========================================================
# Device Setup
# =========================================================

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_CUDA = device.type == "cuda"
IS_CPU  = device.type == "cpu"

# Seed for reproducibility — set before any model init or data shuffling
torch.manual_seed(1337)
if IS_CUDA:
    torch.cuda.manual_seed_all(1337)

n_threads = 0

if IS_CPU:
    if HAS_PSUTIL:
        physical_cores = list(range(psutil.cpu_count(logical=False)))
        psutil.Process().cpu_affinity(physical_cores)
        n_threads = len(physical_cores)
        detail(f"CPU: pinned to {n_threads} physical cores")
    else:
        n_threads = os.cpu_count() or 1
        detail(f"CPU: using {n_threads} logical threads")
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(n_threads)
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
else:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32        = True
    torch.backends.cudnn.benchmark         = True
    detail(f"GPU: {torch.cuda.get_device_name(0)}")

torch.set_float32_matmul_precision("high")

# =========================================================
# Mixed Precision
# =========================================================

if IS_CUDA:
    AMP_DTYPE  = torch.float16
    AMP_DEVICE = "cuda"
    USE_SCALER = True
    scaler     = torch.cuda.amp.GradScaler()
    detail("Mixed precision: float16 (CUDA)")
else:
    def _benchmark_dtype(dtype, steps=5):
        try:
            a = torch.randn(256, 256).to(dtype)
            b = torch.randn(256, 256).to(dtype)
            for _ in range(2): torch.matmul(a, b)
            t0 = time.perf_counter()
            for _ in range(steps): torch.matmul(a, b)
            return (time.perf_counter() - t0) / steps
        except Exception:
            return float("inf")

    t_f32  = _benchmark_dtype(torch.float32)
    t_bf16 = _benchmark_dtype(torch.bfloat16)
    detail(f"dtype benchmark — float32: {t_f32*1000:.3f}ms | bfloat16: {t_bf16*1000:.3f}ms")
    AMP_DTYPE  = torch.bfloat16 if t_bf16 < t_f32 else torch.float32
    AMP_DEVICE = "cpu"
    USE_SCALER = False
    scaler     = None
    detail(f"Mixed precision: {AMP_DTYPE} (CPU)")

def autocast():
    if AMP_DTYPE == torch.float32:
        return torch.autocast(AMP_DEVICE, enabled=False)
    return torch.autocast(AMP_DEVICE, dtype=AMP_DTYPE)

# =========================================================
# Pre-Batched Dataset  (with validation split)
#
# Chunks are non-overlapping slices of the tokenised corpus.
# They are shuffled BEFORE the train/val split so the val set
# is a random sample, not just the tail of the corpus.
# The full dataset tensor is cached to disk keyed by a hash
# of the source files so it is rebuilt automatically when the
# training data changes.
# =========================================================

VAL_SPLIT  = getattr(sys.modules["config"], "VAL_SPLIT", 0.05)
LOG_EVERY  = getattr(sys.modules["config"], "LOG_EVERY",  10)

# Read gradient accumulation steps from config, default to 1
GRAD_ACCUM_STEPS = getattr(sys.modules["config"], "GRAD_ACCUM_STEPS", 1)


class PreBatchedDataset:
    def __init__(self, folder_path, block_size, batch_size):
        self.block_size = block_size
        self.batch_size = batch_size

        data_hash  = hash_data_dir(folder_path)
        cache_name = f"_prebatched_bs{block_size}_{data_hash}.pt"
        cache_path = os.path.join(folder_path, cache_name)

        if os.path.exists(cache_path):
            detail("Loading pre-batched cache from disk...")
            saved    = torch.load(cache_path, map_location="cpu", weights_only=True)
            x, y     = saved["x"], saved["y"]
            n_chunks = saved["n_chunks"]
        else:
            detail("Building pre-batched dataset (caching for future runs)...")
            data = ""
            for root, _, files in os.walk(folder_path):
                for fname in sorted(files):
                    if fname.endswith(".txt") and not fname.startswith("_"):
                        with open(os.path.join(root, fname), "r", encoding="utf-8") as f:
                            data += f.read() + "\n"

            if not data.strip():
                raise ValueError(f"No text found in {folder_path}")

            tokens   = tokenizer.encode(data)
            tokens   = torch.tensor(tokens, dtype=torch.long)
            n_chunks = (len(tokens) - 1) // block_size

            if n_chunks == 0:
                raise ValueError(
                    "Dataset too small for block_size — add more text or reduce BLOCK_SIZE."
                )

            t = tokens[: n_chunks * block_size + 1]
            x = torch.stack([t[i*block_size     : i*block_size + block_size    ] for i in range(n_chunks)])
            y = torch.stack([t[i*block_size + 1 : i*block_size + block_size + 1] for i in range(n_chunks)])

            # Shuffle BEFORE splitting so val set is a random sample
            perm = torch.randperm(n_chunks)
            x, y = x[perm], y[perm]

            torch.save({"x": x, "y": y, "n_chunks": n_chunks}, cache_path)
            detail(f"Cached {n_chunks:,} non-overlapping chunks to {cache_path}")

        # Carve out held-out validation split
        n_val   = max(1, int(n_chunks * VAL_SPLIT))
        n_train = n_chunks - n_val

        self.x_train = x[:n_train]
        self.y_train = y[:n_train]
        self.x_val   = x[n_train:]
        self.y_val   = y[n_train:]

        self.n_train       = n_train
        self.n_val         = n_val
        self.n_batches     = n_train // batch_size
        self.n_val_batches = max(1, n_val // batch_size)

        detail(
            f"Dataset: {n_train:,} train chunks | {n_val:,} val chunks | "
            f"{self.n_batches:,} train batches/epoch | {self.n_val_batches:,} val batches"
        )

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        """Iterate over the training split, shuffling each epoch if configured."""
        if SHUFFLE_DATA_EACH_EPOCH:
            perm    = torch.randperm(self.n_train)
            x_epoch = self.x_train[perm]
            y_epoch = self.y_train[perm]
        else:
            x_epoch = self.x_train
            y_epoch = self.y_train

        for i in range(self.n_batches):
            s = i * self.batch_size
            e = s + self.batch_size
            yield x_epoch[s:e].to(device), y_epoch[s:e].to(device)

    def val_iter(self):
        """Iterate over the validation split (no shuffle)."""
        for i in range(self.n_val_batches):
            s = i * self.batch_size
            e = s + self.batch_size
            yield self.x_val[s:e].to(device), self.y_val[s:e].to(device)


# =========================================================
# Validation loss
# =========================================================

@torch.no_grad()
def evaluate_val_loss(model, dataloader):
    """Mean cross-entropy loss over the full validation split."""
    model.eval()
    total, count = 0.0, 0
    for x, y in dataloader.val_iter():
        with autocast():
            _, loss = model(x, y)
        total += loss.item()
        count += 1
    model.train()
    return total / count if count > 0 else float("nan")


# =========================================================
# Progress / CSV logs
# =========================================================

def write_loss_resp(epoch, avg_loss, val_loss, lr, tok_per_sec, model):
    """Append one CSV row per epoch to loss-resp.log."""
    model.eval()
    responses = []
    with torch.no_grad():
        for prompt in EVAL_PROMPTS:
            try:
                r = generate_response(
                    prompt, model, tokenizer, device, autocast,
                    max_new_tokens=MAX_GEN_TOKENS, stream=False,
                )
                responses.append(r.strip().replace("\n", " "))
            except Exception as e:
                responses.append(f"[error: {e}]")
    # Leave model in eval — caller is responsible for setting train mode
    write_header = (
        not os.path.exists(LOSS_RESP_LOG)
        or os.path.getsize(LOSS_RESP_LOG) == 0
    )

    with open(LOSS_RESP_LOG, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            header = (
                ["epoch", "train_loss", "val_loss", "lr", "tok_per_sec"]
                + [f"response_{i+1}" for i in range(len(EVAL_PROMPTS))]
            )
            writer.writerow(header)
        writer.writerow(
            [epoch, f"{avg_loss:.4f}", f"{val_loss:.4f}", f"{lr:.2e}", tok_per_sec]
            + responses
        )


def write_progress(epoch, avg_loss, val_loss, lr, tok_per_sec, model):
    """Append a human-readable epoch summary to progress.log."""
    model.eval()
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(
            f"Epoch {epoch:>3} | train {avg_loss:.4f} | val {val_loss:.4f} | "
            f"lr {lr:.2e} | {tok_per_sec:,} tok/s\n"
        )
        f.write(f"{'='*60}\n")
        with torch.no_grad():
            for prompt in EVAL_PROMPTS:
                try:
                    response = generate_response(
                        prompt, model, tokenizer, device, autocast,
                        max_new_tokens=MAX_GEN_TOKENS, stream=False,
                    )
                    f.write(f"  > {prompt}\n")
                    f.write(f"    {response.strip()}\n\n")
                except Exception as e:
                    f.write(f"  > {prompt}\n")
                    f.write(f"    [generation failed: {e}]\n\n")
    # Leave model in eval — caller is responsible for setting train mode


# =========================================================
# DataLoader + Model
# =========================================================

def unwrap_model(m):
    """Strip torch.compile wrapper if present."""
    return getattr(m, "_orig_mod", m)

dataloader = PreBatchedDataset("data/training_data/", BLOCK_SIZE, BATCH_SIZE)
if len(dataloader) == 0:
    error("Dataset produced 0 batches — add more text or reduce BLOCK_SIZE/BATCH_SIZE.")
    sys.exit(1)

model = GPT(GPTConfig(vocab_size=vocab_size, block_size=BLOCK_SIZE)).to(device)

param_count = sum(p.numel() for p in model.parameters())
detail(f"Model parameters: {param_count:,}")

if IS_CUDA and hasattr(torch, "compile"):
    model = torch.compile(model, mode="reduce-overhead")
    detail("torch.compile: reduce-overhead [GPU]")
else:
    detail("torch.compile: disabled [CPU]")

# =========================================================
# Optimizer
# =========================================================

# Support both single LR (original config) and LR_MAX/LR_MIN (advanced config)
try:
    from config import LEARNING_RATE_MAX, LEARNING_RATE_MIN
    LR_MAX = LEARNING_RATE_MAX
    LR_MIN = LEARNING_RATE_MIN
except ImportError:
    LR_MAX = LEARNING_RATE
    LR_MIN = LEARNING_RATE * 0.01

BETA_ONE = getattr(sys.modules["config"], "BETA_ONE", 0.9)
BETA_TWO = getattr(sys.modules["config"], "BETA_TWO", 0.95)

if IS_CPU:
    # Lion can behave poorly on CPU without proper bfloat16 support;
    # AdamW with foreach=True is faster and more stable on CPU.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR_MAX,
        weight_decay=WEIGHT_DECAY,
        betas=(BETA_ONE, BETA_TWO),
        foreach=True,
    )
    detail("Optimizer: AdamW foreach=True [CPU]")
else:
    try:
        from lion_pytorch import Lion
        optimizer = Lion(
            model.parameters(),
            lr=LR_MAX,
            weight_decay=WEIGHT_DECAY,
            betas=(BETA_ONE, BETA_TWO),
        )
        detail("Optimizer: Lion [GPU]")
    except ImportError:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR_MAX,
            weight_decay=WEIGHT_DECAY,
            betas=(BETA_ONE, BETA_TWO),
            foreach=True,
        )
        detail("Optimizer: AdamW foreach=True (lion_pytorch not found) [GPU]")

# =========================================================
# LR Scheduler
#
# total_steps is based on gradient-update steps
# (batches / GRAD_ACCUM_STEPS), not raw batch count.
# This matters when GRAD_ACCUM_STEPS > 1 — previously the
# scheduler decayed too slowly, keeping LR too high.
# =========================================================

total_steps  = (EPOCHS * len(dataloader)) // GRAD_ACCUM_STEPS
warmup_steps = int(getattr(sys.modules["config"], "SCHEDULER_WARMUP", 0.1) * total_steps)
sched_type   = getattr(sys.modules["config"], "SCHEDULER_TYPE", "cosine")

if warmup_steps > 0 and sched_type == "cosine":
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=LR_MIN / LR_MAX, end_factor=1.0,
                     total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps),
                              eta_min=LR_MIN),
        ],
        milestones=[warmup_steps],
    )
    detail(f"Scheduler: warmup {warmup_steps} steps then cosine {LR_MAX:.2e} -> {LR_MIN:.2e}")
elif sched_type == "cosine":
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=LR_MIN)
    detail(f"Scheduler: cosine {LR_MAX:.2e} -> {LR_MIN:.2e} over {total_steps} steps")
elif sched_type == "linear":
    scheduler = LinearLR(optimizer, start_factor=1.0,
                         end_factor=LR_MIN / LR_MAX, total_iters=total_steps)
    detail(f"Scheduler: linear {LR_MAX:.2e} -> {LR_MIN:.2e} over {total_steps} steps")
else:
    # Flat — constant LR
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=LR_MAX)
    detail(f"Scheduler: flat {LR_MAX:.2e}")

# =========================================================
# Checkpoints
# =========================================================

checkpoint_path  = "data/models/model.pt"
epoch_path       = "data/models/last_epoch.txt"
scheduler_path   = "data/models/scheduler.pt"
progress_path    = "data/models/progress.pt"   # FIX: saves batches_done for accurate resume
start_epoch      = 0
resumed_batches  = 0  # FIX: track actual completed batches across resumed runs

if os.path.exists(checkpoint_path):
    ans = input("Checkpoint found. Resume? (y/n): ").strip().lower()
    if ans == "y":
        unwrap_model(model).load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
        detail("Resumed model weights.")
        if os.path.exists(epoch_path):
            with open(epoch_path) as f:
                start_epoch = int(f.read().strip())
        if os.path.exists(scheduler_path):
            scheduler.load_state_dict(
                torch.load(scheduler_path, map_location="cpu", weights_only=True)
            )
            detail(
                f"Resumed scheduler | last_epoch={scheduler.last_epoch} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
        # FIX: restore batches_done so ETA/tok-per-sec are meaningful after resume
        if os.path.exists(progress_path):
            prog = torch.load(progress_path, map_location="cpu", weights_only=True)
            resumed_batches = prog.get("batches_done", 0)
            detail(f"Resumed progress: {resumed_batches:,} batches previously done")
        info(f"Resuming from epoch {start_epoch + 1}")

# =========================================================
# Startup summary
# =========================================================

dtype_name = {
    torch.float32:  "float32",
    torch.float16:  "float16",
    torch.bfloat16: "bfloat16",
}[AMP_DTYPE]
device_str = (
    f"GPU ({torch.cuda.get_device_name(0)})" if IS_CUDA
    else f"CPU ({n_threads} cores)"
)

info(f"  Device   : {device_str}")
info(f"  Dtype    : {dtype_name}")
info(f"  Params   : {param_count:,}")
info(f"  Chunks   : {dataloader.n_train:,} train | {dataloader.n_val:,} val")
info(f"  Batches  : {len(dataloader):,}/epoch")
info(f"  Epochs   : {EPOCHS}  |  Batch: {BATCH_SIZE}  |  Block: {BLOCK_SIZE}")
info(f"  LR       : {LR_MAX:.2e} -> {LR_MIN:.2e} ({sched_type})")
info(f"  Accum    : {GRAD_ACCUM_STEPS} steps")
info(f"  Val split: {VAL_SPLIT:.0%}")
info(f"  Log file : training.log")
info(f"  Progress : {PROGRESS_LOG}")
info(f"  CSV log  : {LOSS_RESP_LOG}")
info("")

# =========================================================
# Utilities
# =========================================================

def cleanup():
    if IS_CUDA:
        torch.cuda.empty_cache()
    gc.collect()

_shutting_down = False

def signal_handler(signum, frame):
    global _shutting_down
    if _shutting_down:
        return
    _shutting_down = True
    print("\nStopping...")
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT,  signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# =========================================================
# Profiler mode  (PROFILE=1 python train.py)
# =========================================================

if os.environ.get("PROFILE", "0") == "1":
    from torch.profiler import profile, ProfilerActivity
    info("[PROFILER] Running 20 steps...")
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        for step, (x, y) in enumerate(dataloader):
            if step >= 20:
                break
            with autocast():
                _, loss = model(x, y)
            # FIX: use properly scaled loss in profiler so backward time is realistic
            loss = loss / GRAD_ACCUM_STEPS
            if USE_SCALER:
                scaler.scale(loss).backward()
            else:
                loss.backward()
    table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=15)
    print(table)
    detail(table)
    sys.exit(0)

# =========================================================
# Training Loop
# =========================================================

start_time    = time.time()
# FIX: start from resumed_batches so elapsed-time calculations survive resume
batches_done  = resumed_batches
total_batches = EPOCHS * len(dataloader)

def save_best():
    pass  # used as a namespace to carry best_val across epochs

for epoch in range(start_epoch, EPOCHS):
    model.train()
    loop = tqdm(
        dataloader,
        desc  = f"Epoch {epoch + 1}/{EPOCHS}",
        ncols = None,
        leave = True,
        total = len(dataloader),
    )

    total_loss  = 0.0
    batch_count = 0
    # FIX: track whether the final accumulation window in the epoch was already stepped,
    # using a dedicated flag that isn't reset inside the per-step loop.
    last_accum_stepped = False
    optimizer.zero_grad(set_to_none=True)

    for step, (x, y) in enumerate(loop):
        with autocast():
            logits, loss = model(x, y)

        # Detect exploding / vanishing gradients early
        if torch.isnan(loss) or torch.isinf(loss):
            error(f"Loss is NaN/Inf at step {step}. Stopping.")
            sys.exit(1)

        true_loss = loss.item()
        # Scale loss for gradient accumulation
        loss = loss / GRAD_ACCUM_STEPS

        if USE_SCALER:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # FIX: track whether this step completed an accumulation window
        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            if USE_SCALER:
                scaler.unscale_(optimizer)
                if GRAD_CLIP:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                if GRAD_CLIP:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            last_accum_stepped = True
        else:
            last_accum_stepped = False

        total_loss  += true_loss
        batch_count += 1
        batches_done += 1

        if batch_count % LOG_EVERY == 0:
            avg_loss   = total_loss / batch_count
            elapsed    = time.time() - start_time
            # FIX: ETA uses batches done this session only (start_time is this session)
            batches_this_session = batches_done - resumed_batches
            remaining  = (
                (total_batches - batches_done) * (elapsed / batches_this_session)
                if batches_this_session > 0 and elapsed > 0 else 0
            )
            tok_per_sec = int(batches_this_session * BATCH_SIZE * BLOCK_SIZE / elapsed) if elapsed > 0 else 0
            mins, secs  = divmod(remaining, 60)
            current_lr  = scheduler.get_last_lr()[0]
            loop.set_postfix(
                loss = f"{avg_loss:.4f}",
                lr   = f"{current_lr:.2e}",
                tok  = f"{tok_per_sec:,}",
                eta  = f"{int(mins):02d}:{int(secs):02d}",
            )

        if batch_count % 50 == 0:
            detail(
                f"Epoch {epoch+1} step {batch_count} | "
                f"loss {total_loss/batch_count:.4f} | "
                f"lr {scheduler.get_last_lr()[0]:.2e}"
            )

    # FIX: only handle leftover gradients if the final batch did NOT complete
    # an accumulation window — prevents the double scheduler.step() bug.
    if not last_accum_stepped:
        if USE_SCALER:
            scaler.unscale_(optimizer)
            if GRAD_CLIP:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            if GRAD_CLIP:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    elapsed     = time.time() - start_time
    avg_loss    = total_loss / batch_count
    batches_this_session = batches_done - resumed_batches
    tok_per_sec = int(batches_this_session * BATCH_SIZE * BLOCK_SIZE / elapsed) if elapsed > 0 else 0
    # FIX: capture LR after all end-of-epoch stepping is done (moved to after leftover block)
    current_lr  = scheduler.get_last_lr()[0]

    # Evaluate validation loss with model in eval mode, no grad
    val_loss = evaluate_val_loss(model, dataloader)

    info(
        f"Epoch {epoch + 1:>3}/{EPOCHS} | "
        f"train {avg_loss:.4f} | val {val_loss:.4f} | "
        f"lr {current_lr:.2e} | {tok_per_sec:,} tok/s"
    )

    # Save checkpoints
    epoch_dir = f"data/models/epoch{epoch + 1}"
    os.makedirs(epoch_dir, exist_ok=True)
    torch.save(unwrap_model(model).state_dict(), f"{epoch_dir}/model.pt")
    torch.save(unwrap_model(model).state_dict(), checkpoint_path)
    torch.save(scheduler.state_dict(), scheduler_path)
    with open(epoch_path, "w") as f:
        f.write(str(epoch + 1))
    # FIX: persist batches_done so resume can restore accurate progress tracking
    torch.save({"batches_done": batches_done}, progress_path)
    detail(f"Checkpoint saved: {epoch_dir}/model.pt")

    # Save best-val-loss checkpoint separately so it survives overfitting epochs
    best_val_path = "data/models/model_best.pt"
    if not hasattr(save_best, "best_val") or val_loss < save_best.best_val:
        save_best.best_val = val_loss
        torch.save(unwrap_model(model).state_dict(), best_val_path)
        detail(f"New best val loss {val_loss:.4f} — saved to {best_val_path}")

    # Terminal generation preview — eval mode + no_grad
    model.eval()
    try:
        with torch.no_grad():
            preview = generate_response(
                EVAL_PROMPTS[0], model, tokenizer, device, autocast,
                max_new_tokens=MAX_GEN_TOKENS, stream=False,
            )
        info(f"         sample: {preview[:80].replace(chr(10), ' ')}")
    except Exception as e:
        warn(f"Generation failed: {e}")

    # FIX: model is already in eval mode from the preview block above.
    # write_progress and write_loss_resp expect eval mode and leave model in eval.
    # We set train mode once at the end instead of ping-ponging inside each function.
    write_progress(epoch + 1, avg_loss, val_loss, current_lr, tok_per_sec, model)
    write_loss_resp(epoch + 1, avg_loss, val_loss, current_lr, tok_per_sec, model)

    # Return to train mode for next epoch
    model.train()

    cleanup()

info("")
info("Training complete.")
input("Press Enter to exit...")