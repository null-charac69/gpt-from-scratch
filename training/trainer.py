
"""
Trainer class - handles the training loop, optimization, and evaluation.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import os
import sys

# Modern PyTorch AMP support with backwards compatibility
try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler
    def get_autocast(device_type):
        return _autocast(device_type=device_type)
    def get_scaler(device_type):
        return _GradScaler(device_type=device_type)
except (ImportError, AttributeError):
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler
    def get_autocast(device_type):
        return _autocast()
    def get_scaler(device_type):
        return _GradScaler()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import (
    get_lr, save_checkpoint, estimate_loss, 
    AverageMeter, Timer, clip_gradients, log_metrics
)


class Trainer:
    """
    Trainer for GPT model.
    Handles training loop, optimization, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        config,
        run_dir='runs',
        use_amp=False
    ):
        """
        Args:
            model: GPT model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer (e.g., AdamW)
            device: Device to train on
            config: TrainingConfig instance
            run_dir: Directory to save checkpoints and logs
            use_amp: Whether to use automatic mixed precision
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.run_dir = run_dir
        self.use_amp = use_amp and device.type == 'cuda'
        
        # Automatic mixed precision
        self.scaler = get_scaler(self.device.type) if self.use_amp else None
        
        # Tracking
        self.iteration = 0
        self.best_val_loss = float('inf')
        
        # Create directories
        os.makedirs(run_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(run_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Logging
        self.log_file = os.path.join(run_dir, 'train.log')
    
    def train_step(self, x, y):
        """
        Single training step.
        
        Args:
            x: Input tokens
            y: Target tokens
            
        Returns:
            Loss value
        """
        # Forward pass
        if self.use_amp:
            with get_autocast(self.device.type):
                logits, loss = self.model(x, targets=y)
        else:
            logits, loss = self.model(x, targets=y)
        
        # Backward pass
        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def optimizer_step(self):
        """
        Perform optimizer step with gradient clipping.
        """
        # Clip gradients
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
        
        grad_norm = clip_gradients(self.model, self.config.max_grad_norm)
        
        # Optimizer step
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad(set_to_none=True)
        
        return grad_norm
    
    def train(self):
        """
        Main training loop.
        """
        print(f"Starting training for {self.config.max_iters} iterations...")
        print(f"Checkpoints will be saved to {self.checkpoint_dir}")
        
        self.model.train()
        
        # Metrics
        loss_meter = AverageMeter()
        timer = Timer()
        
        # Progress bar
        pbar = tqdm(range(self.config.max_iters), desc="Training")
        
        for iter_num in pbar:
            self.iteration = iter_num
            
            # Get learning rate for this iteration
            lr = get_lr(
                iter_num,
                self.config.warmup_iters,
                self.config.learning_rate,
                self.config.lr_decay_iters,
                self.config.min_lr
            )
            
            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Training step with gradient accumulation
            timer.start()
            for micro_step in range(self.config.gradient_accumulation_steps):
                x, y = next(self.train_loader)
                x, y = x.to(self.device), y.to(self.device)
                
                loss = self.train_step(x, y)
                loss_meter.update(loss)
            
            # Optimizer step
            grad_norm = self.optimizer_step()
            dt = timer.stop()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'lr': f'{lr:.2e}',
                'ms/iter': f'{dt*1000:.1f}'
            })
            
            # Evaluation
            if (iter_num + 1) % self.config.eval_interval == 0:
                val_loss = self.evaluate()
                
                # Log metrics
                metrics = {
                    'iteration': iter_num + 1,
                    'train_loss': loss_meter.avg,
                    'val_loss': val_loss,
                    'lr': lr,
                    'grad_norm': grad_norm,
                }
                log_metrics(metrics, self.log_file)
                
                # Save checkpoint
                self.save_checkpoint(val_loss)
                
                # Reset meters
                loss_meter.reset()
                
                self.model.train()
        
        print("Training complete!")
        return self.best_val_loss
    
    def evaluate(self):
        """
        Evaluate model on validation set.
        
        Returns:
            Validation loss
        """
        if self.val_loader is None:
            return 0.0
        
        val_loss = estimate_loss(
            self.model,
            self.val_loader,
            self.config.eval_iters,
            self.device
        )
        
        print(f"\nValidation loss: {val_loss:.4f}")
        return val_loss
    
    def save_checkpoint(self, val_loss):
        """
        Save model checkpoint.
        
        Args:
            val_loss: Validation loss for this checkpoint
        """
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, 'latest.pt')
        save_checkpoint(
            self.model,
            self.optimizer,
            self.iteration,
            val_loss,
            checkpoint_path
        )
        
        # Save best checkpoint
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(self.checkpoint_dir, 'best.pt')
            save_checkpoint(
                self.model,
                self.optimizer,
                self.iteration,
                val_loss,
                best_path
            )
            print(f"New best model! Validation loss: {val_loss:.4f}")
        
        # Save periodic checkpoint
        if (self.iteration + 1) % (self.config.eval_interval * 10) == 0:
            periodic_path = os.path.join(
                self.checkpoint_dir,
                f'checkpoint_iter_{self.iteration+1}.pt'
            )
            save_checkpoint(
                self.model,
                self.optimizer,
                self.iteration,
                val_loss,
                periodic_path
            )


class SimpleTrainer:
    """
    Simplified trainer for quick experiments.
    Minimal logging and evaluation.
    """
    
    def __init__(self, model, train_loader, optimizer, device, max_iters=1000):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device
        self.max_iters = max_iters
    
    def train(self):
        """Simple training loop."""
        self.model.train()
        
        pbar = tqdm(range(self.max_iters), desc="Training")
        for i in pbar:
            x, y = next(self.train_loader)
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward
            logits, loss = self.model(x, targets=y)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Update progress
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        print("Training complete!")


def configure_optimizers(model, config):
    """
    Configure AdamW optimizer with weight decay.
    Separates parameters into decay and no-decay groups.
    
    Args:
        model: Model to optimize
        config: TrainingConfig
        
    Returns:
        Configured optimizer
    """
    # Separate parameters into weight decay and no weight decay groups
    decay = set()
    no_decay = set()
    
    # All 2D parameters (weights) should have weight decay
    # All 1D parameters (biases, layer norms) should not
    whitelist_weight_modules = (nn.Linear, )
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
    
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f'{mn}.{pn}' if mn else pn
            
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)
    
    # Special case for positional embeddings
    no_decay.add('position_embedding.position_embeddings.weight')
    
    # Validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"Parameters in both decay and no_decay: {inter_params}"
    assert len(param_dict.keys() - union_params) == 0, \
        f"Parameters not in either set: {param_dict.keys() - union_params}"
    
    # Create optimizer groups
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": config.weight_decay
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0
        },
    ]
    
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2)
    )
    
    return optimizer