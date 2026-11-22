"""
Main training script for GPT.
Run this to train a GPT model from scratch.

Usage:
    python train.py
"""

import torch
import argparse
import os

from model.gpt import GPT
from training.config import GPTConfig, TrainingConfig
from training.trainer import Trainer, configure_optimizers
from data.tokenizer import SimpleTokenizer
from data.dataset import DataLoaderWrapper, prepare_sample_data, download_shakespeare
from utils.helpers import set_seed, get_device, print_model_summary, create_run_directory


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GPT from scratch')
    
    # Model args
    parser.add_argument('--n_layer', type=int, default=6, help='Number of layers')
    parser.add_argument('--n_head', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=384, help='Embedding dimension')
    parser.add_argument('--n_positions', type=int, default=256, help='Context length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training args
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--max_iters', type=int, default=5000, help='Max training iterations')
    parser.add_argument('--eval_interval', type=int, default=100, help='Evaluation interval')
    parser.add_argument('--warmup_iters', type=int, default=100, help='Warmup iterations')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, 
                       help='Gradient accumulation steps')
    
    # Data args
    parser.add_argument('--dataset', type=str, default='sample', 
                       choices=['sample', 'shakespeare'],
                       help='Dataset to use')
    
    # System args
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--compile', action='store_true', help='Compile model (PyTorch 2.0+)')
    parser.add_argument('--run_dir', type=str, default='runs', help='Directory for runs')
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create run directory
    run_dir = create_run_directory(args.run_dir)
    print(f"Run directory: {run_dir}")
    
    # Load data
    print("\nLoading data...")
    if args.dataset == 'shakespeare':
        text_file = download_shakespeare()
        with open(text_file, 'r') as f:
            text = f.read()
        # Split train/val
        split_idx = int(len(text) * 0.9)
        train_text = text[:split_idx]
        val_text = text[split_idx:]
    else:
        train_text, val_text = prepare_sample_data()
    
    print(f"Train text length: {len(train_text)} characters")
    print(f"Validation text length: {len(val_text)} characters")
    
    # Create tokenizer
    print("\nCreating tokenizer...")
    tokenizer = SimpleTokenizer()
    tokenizer.build_from_text(train_text + val_text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create model config
    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_positions=args.n_positions,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dropout=args.dropout,
    )
    
    # Create training config
    train_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        warmup_iters=args.warmup_iters,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    # Create data loaders
    print("\nCreating data loaders...")
    data_wrapper = DataLoaderWrapper(train_text, val_text, tokenizer, model_config)
    train_loader = data_wrapper.get_train_loader(
        batch_size=train_config.batch_size,
        infinite=True
    )
    val_loader = data_wrapper.get_val_loader(batch_size=train_config.batch_size)
    
    # Create model
    print("\nCreating model...")
    model = GPT(model_config)
    model = model.to(device)
    
    # Print model summary
    print_model_summary(model, model_config)
    
    # Compile model (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model...")
        model = torch.compile(model)
    
    # Create optimizer
    print("Creating optimizer...")
    optimizer = configure_optimizers(model, train_config)
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        config=train_config,
        run_dir=run_dir,
        use_amp=args.use_amp
    )
    
    # Train
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50 + "\n")
    
    best_val_loss = trainer.train()
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {trainer.checkpoint_dir}")
    
    # Save final model
    final_path = os.path.join(run_dir, 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config.__dict__,
        'tokenizer_chars': tokenizer.chars,
    }, final_path)
    print(f"Final model saved to: {final_path}")


if __name__ == '__main__':
    main()