"""
Configuration file for GPT model hyperparameters.
This defines the architecture and training settings.
"""

class GPTConfig:
    """Configuration class for GPT model."""
    
    def __init__(
        self,
        vocab_size=50257,          # Size of vocabulary (default: GPT-2 vocab size)
        n_positions=1024,          # Maximum sequence length
        n_embd=768,                # Embedding dimension
        n_layer=12,                # Number of transformer layers
        n_head=12,                 # Number of attention heads
        n_inner=None,              # Inner dimension of FFN (4 * n_embd if None)
        dropout=0.1,               # Dropout probability
        layer_norm_epsilon=1e-5,   # Epsilon for layer normalization
        initializer_range=0.02,    # Standard deviation for weight initialization
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner if n_inner is not None else 4 * n_embd
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        
        # Derived attributes
        self.head_dim = n_embd // n_head
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"


class TrainingConfig:
    """Configuration for training hyperparameters."""
    
    def __init__(
        self,
        batch_size=8,
        learning_rate=3e-4,
        max_iters=5000,
        eval_interval=100,
        eval_iters=20,
        warmup_iters=100,
        lr_decay_iters=5000,
        min_lr=3e-5,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2


# Preset configurations for different model sizes
def get_gpt2_config(model_type='gpt2'):
    """
    Returns configuration for different GPT-2 model sizes.
    
    Args:
        model_type: One of 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
    """
    configs = {
        'gpt2': dict(n_layer=12, n_head=12, n_embd=768),           # 117M params
        'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),   # 345M params
        'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),    # 774M params
        'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),       # 1558M params
    }
    
    if model_type not in configs:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return GPTConfig(**configs[model_type])