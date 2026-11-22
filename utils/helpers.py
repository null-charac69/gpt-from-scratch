"""
Utility helper functions for training and evaluation.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
import json


def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_device():
    """
    Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(it, warmup_iters, learning_rate, lr_decay_iters, min_lr):
    """
    Learning rate schedule with warmup and cosine decay.
    
    Args:
        it: Current iteration
        warmup_iters: Number of warmup iterations
        learning_rate: Maximum learning rate
        lr_decay_iters: Number of iterations for decay
        min_lr: Minimum learning rate
        
    Returns:
        Current learning rate
    """
    # Linear warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    
    # After decay period, return min lr
    if it > lr_decay_iters:
        return min_lr
    
    # Cosine decay
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def save_checkpoint(model, optimizer, iteration, loss, path):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        iteration: Current iteration
        loss: Current loss
        path: Path to save checkpoint
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, device):
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        path: Path to checkpoint
        device: Device to load tensors to
        
    Returns:
        Tuple of (iteration, loss)
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    iteration = checkpoint.get('iteration', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    print(f"Checkpoint loaded from {path}")
    print(f"Resuming from iteration {iteration}, loss {loss:.4f}")
    
    return iteration, loss


def estimate_loss(model, data_loader, eval_iters, device):
    """
    Estimate loss on a dataset.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        eval_iters: Number of iterations to evaluate
        device: Device to run on
        
    Returns:
        Average loss
    """
    model.eval()
    losses = []
    
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            if i >= eval_iters:
                break
            
            x = x.to(device)
            y = y.to(device)
            
            _, loss = model(x, targets=y)
            losses.append(loss.item())
    
    model.train()
    return np.mean(losses)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics during training.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer:
    """Simple timer for measuring execution time."""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        if self.start_time is not None:
            self.elapsed = time.time() - self.start_time
            self.start_time = None
        return self.elapsed
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


def format_time(seconds):
    """
    Format seconds into human-readable time.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_model_summary(model, config):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        config: Model configuration
    """
    print("\n" + "="*50)
    print("MODEL SUMMARY")
    print("="*50)
    print(f"Model type: GPT")
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Context length: {config.n_positions}")
    print(f"Embedding dimension: {config.n_embd}")
    print(f"Number of layers: {config.n_layer}")
    print(f"Number of heads: {config.n_head}")
    print(f"FFN dimension: {config.n_inner}")
    print(f"Dropout: {config.dropout}")
    print(f"\nTotal parameters: {count_parameters(model):,}")
    print("="*50 + "\n")


def log_metrics(metrics, log_file=None):
    """
    Log training metrics.
    
    Args:
        metrics: Dictionary of metrics
        log_file: Optional file to append logs
    """
    # Print to console
    metric_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                             for k, v in metrics.items()])
    print(metric_str)
    
    # Append to log file
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')


def create_run_directory(base_dir='runs'):
    """
    Create a directory for this training run with timestamp.
    
    Args:
        base_dir: Base directory for runs
        
    Returns:
        Path to created directory
    """
    os.makedirs(base_dir, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_dir, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir


def clip_gradients(model, max_norm):
    """
    Clip gradients by global norm.
    
    Args:
        model: Model whose gradients to clip
        max_norm: Maximum gradient norm
        
    Returns:
        Total gradient norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def get_batch_from_dataset(data_loader):
    """
    Get a single batch from data loader.
    Handles StopIteration by recreating iterator.
    
    Args:
        data_loader: PyTorch DataLoader
        
    Returns:
        Batch (x, y)
    """
    try:
        return next(data_loader)
    except StopIteration:
        return next(iter(data_loader))