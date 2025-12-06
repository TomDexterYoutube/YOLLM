# === Model & Training Configuration ===

BLOCK_SIZE = 128  # Number of tokens in each input sequence (context window size)
BATCH_SIZE = 3000  # Number of samples per training batch
LEARNING_RATE = 0.001  # Learning rate for the optimizer
EPOCHS = 100  # Number of full passes through the training dataset
DROPOUT = 0.20  # Dropout probability for regularization in the model
HIDDEN_DIM = 30  # Increased from 200 for more capacity
EMBEDDING_DIM = HIDDEN_DIM # Embedding dimension for token and position embeddings (usually same as HIDDEN_DIM)
FEEDFORWARD_MULTIPLIER = 4  # Increased from 4 for more complex representations
N_LAYERS = 4  # Increased from 4 for more depth
N_HEADS = N_LAYERS / 2  # Number of attention heads in each transformer block
MAX_GEN_TOKENS = 120  # Maximum number of tokens to generate during inference
MAX_TRAIN_TOKENS = 64  # Maximum number of tokens to generate for evaluation during training
DATA_PATH = "data/training_data/"  # Directory containing your training .txt files
SIMILARITY_SENSITIVITY = 0.6

# === Tokenizer Configuration ===
TOKENIZER_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 !@#$%^&*()-_+={[}]:;\"'<,>/?.~`"  # Allowed characters for the tokenizer's vocabulary (letters, numbers, punctuation, symbols, and space)
TOKENIZER_MIN_FREQUENCY = 1  # Minimum number of occurrences for a token to be included in the vocabulary
TOKENIZER_VOCAB_SIZE = 0  # or 256, 128, Set to an integer or "auto" to use the number of unique tokens in the cleaned dataset

# === Training Options ===
WEIGHT_DECAY = 0.01  # Weight decay (L2 regularization) for the optimizer
GRAD_CLIP = 1.0  # Maximum norm for gradient clipping (set to None to disable)
SCHEDULER_TYPE = "cosine"  # Type of learning rate scheduler: 'cosine', 'step', or None
SCHEDULER_WARMUP = 0.1  # Fraction of total epochs to use for learning rate warmup

# === Generation Options ===
TEMPERATURE = 0.5  # Sampling temperature (higher = more random)
TOP_P = 0.9  # Nucleus sampling threshold
REPETITION_PENALTY = 0.6  # Penalty for repeating tokens
NO_REPEAT_NGRAM_SIZE = 1  # Size of n-grams to prevent repeating
MIN_LENGTH = 1  # Minimum generation length
MAX_LENGTH = MAX_GEN_TOKENS - 1  # Maximum generation length

# === Training Penalties ===
SIMILARITY_PENALTY = 0.6  # Penalty for training data similarity
COMPLEXITY_REWARD = 0.2   # Reward for vocabulary diversity
MIN_UNIQUE_TOKENS = 0    # Required unique tokens
DIVERSITY_FACTOR = 0.4    # Pattern repetition penalty
QUALITY_LOSS_MULTIPLIER = 1.0  # Multiplier based on epoch generation quality (1.0 = no adjustment)

SHUFFLE_DATA_EACH_EPOCH = True  # Whether to shuffle training data at each epoch

# === Word Monitoring Configuration ===
REWARD_WORDS = ["edward:", "hello", "hi", "~"]  # Words to reward (appears = lower loss)
WORD_PENALTY = 0.3  # Loss increase per penalize word found
WORD_REWARD = 0.3  # Loss decrease per reward word found

# === Model Configuration Class ===

class ModelConfig:
    def __init__(self):
        # Use global constants instead of duplicating values
        self.temperature = TEMPERATURE
        self.top_p = TOP_P
        self.repetition_penalty = REPETITION_PENALTY
        self.no_repeat_ngram_size = NO_REPEAT_NGRAM_SIZE
        self.max_length = MAX_LENGTH
        self.min_length = MIN_LENGTH
        self.similarity_penalty = SIMILARITY_PENALTY
        self.complexity_reward = COMPLEXITY_REWARD
        self.min_unique_tokens = MIN_UNIQUE_TOKENS
        self.diversity_factor = DIVERSITY_FACTOR

    def get_generation_params(self):
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "max_length": self.max_length,
            "min_length": self.min_length,
            "similarity_penalty": self.similarity_penalty,
            "complexity_reward": self.complexity_reward,
            "min_unique_tokens": self.min_unique_tokens,
            "diversity_factor": self.diversity_factor
        }
