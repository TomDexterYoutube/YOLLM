# =============================================================================
#  YOLLM — Configuration
#  All tunable settings live here. Everything else is automatic.
# =============================================================================


# ── Model Architecture ────────────────────────────────────────────────────────

HIDDEN_DIM             = 64   # Width of the model. Must be divisible by N_HEADS.
N_LAYERS               = 6     # Number of transformer blocks.
N_HEADS                = 8     # Attention heads. Must divide HIDDEN_DIM evenly.
N_KV_HEADS             = 2     # KV heads for GQA. Must divide N_HEADS evenly.
                               #   N_KV_HEADS == N_HEADS  →  full MHA
                               #   N_KV_HEADS == 1        →  MQA (lightest)
FEEDFORWARD_MULTIPLIER = 4     # FF hidden size = int(HIDDEN_DIM * this * 2/3)
BLOCK_SIZE             = 128   # Context window in tokens (characters).
DROPOUT                = 0.0  # Dropout probability. 0.0 to disable.


# ── Training ──────────────────────────────────────────────────────────────────

EPOCHS             = 250   # Full passes through training data.
BATCH_SIZE         = 64    # Sequences per batch. Increase if you have more RAM.
GRAD_ACCUM_STEPS   = 2     # Effective batch = BATCH_SIZE × this.
WEIGHT_DECAY       = 0.0  # L2 regularisation strength.
GRAD_CLIP          = 1.0   # Max gradient norm. None to disable.
SHUFFLE_DATA_EACH_EPOCH = True


# ── Learning Rate ─────────────────────────────────────────────────────────────

LEARNING_RATE     = 1e-3   # Used directly if scheduler is "flat".
LEARNING_RATE_MAX = 5e-2   # Peak LR reached after warmup.
LEARNING_RATE_MIN = 1e-6   # Floor LR at the end of decay.
SCHEDULER_TYPE    = "linear"  # "cosine" | "linear" | "flat"
SCHEDULER_WARMUP  = 0.0    # Fraction of steps for linear warmup. 0.0 = start high immediately.
BETA_ONE          = 0.9    # Optimizer beta1.
BETA_TWO          = 0.95   # Optimizer beta2.


# ── Validation & Logging ──────────────────────────────────────────────────────

VAL_SPLIT  = 0.05   # Fraction of data held out for validation.
LOG_EVERY  = 1     # Update tqdm progress bar every N batches.


# ── Generation ────────────────────────────────────────────────────────────────

MAX_GEN_TOKENS       = 120   # Max tokens generated at inference.
MAX_TRAIN_TOKENS     = 64    # Max tokens generated during training eval previews.
TEMPERATURE          = 0.8   # Higher = more random. Lower = more repetitive.
TOP_P                = 0.9   # Nucleus sampling. 1.0 = disabled.
REPETITION_PENALTY   = 1.3   # > 1.0 penalises repeated tokens. 1.0 = disabled.
NO_REPEAT_NGRAM_SIZE = 3     # Blocks repeated n-grams. 0 = disabled.


# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_PATH = "data/training_data/"


# ── Internal (do not edit) ────────────────────────────────────────────────────

EMBEDDING_DIM = HIDDEN_DIM   # Always tied to HIDDEN_DIM.
MIN_LENGTH    = 1
MAX_LENGTH    = MAX_GEN_TOKENS - 1
