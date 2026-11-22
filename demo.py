"""
Quick demo script to test GPT implementation.
Trains a tiny model on sample data and generates text.

Usage:
    python demo.py
"""

import torch
from model.gpt import GPT
from training.config import GPTConfig, TrainingConfig
from training.trainer import SimpleTrainer, configure_optimizers
from data.tokenizer import SimpleTokenizer
from data.dataset import TextDataset, InfiniteDataLoader
from utils.helpers import set_seed, get_device

def main():
    print("="*60)
    print("GPT FROM SCRATCH - QUICK DEMO")
    print("="*60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Sample text
    text = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
""" * 50  # Repeat for more data
    
    print(f"\nTraining on {len(text)} characters")
    
    # Create tokenizer
    print("\n1. Creating tokenizer...")
    tokenizer = SimpleTokenizer()
    tokenizer.build_from_text(text)
    print(f"   Vocabulary size: {tokenizer.vocab_size}")
    
    # Create tiny model
    print("\n2. Creating model...")
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_positions=128,
        n_embd=128,
        n_layer=4,
        n_head=4,
        dropout=0.1,
    )
    
    model = GPT(config)
    model = model.to(device)
    print(f"   Model parameters: {model.get_num_params()/1e6:.2f}M")
    
    # Create dataset
    print("\n3. Creating dataset...")
    dataset = TextDataset(text, tokenizer, block_size=128)
    train_loader = InfiniteDataLoader(dataset, batch_size=8)
    print(f"   Dataset size: {len(dataset)} samples")
    
    # Create optimizer
    print("\n4. Creating optimizer...")
    train_config = TrainingConfig(
        learning_rate=1e-3,
        max_iters=500,
    )
    optimizer = configure_optimizers(model, train_config)
    
    # Train
    print("\n5. Training...")
    print("-"*60)
    trainer = SimpleTrainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        max_iters=500
    )
    trainer.train()
    print("-"*60)
    
    # Generate text
    print("\n6. Generating text...")
    print("-"*60)
    
    prompts = [
        "To be",
        "The slings",
        "To die",
    ]
    
    model.eval()
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        
        # Encode prompt
        context = torch.tensor(
            tokenizer.encode(prompt),
            dtype=torch.long,
            device=device
        ).unsqueeze(0)
        
        # Generate
        with torch.no_grad():
            generated = model.generate(
                context,
                max_new_tokens=50,
                temperature=0.8,
                top_k=10,
            )
        
        # Decode and print
        generated_text = tokenizer.decode(generated[0].tolist())
        print(f"Generated: {generated_text}")
    
    print("-"*60)
    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)
    print("\nTo train a full model, run:")
    print("  python train.py --dataset shakespeare --max_iters 5000")
    print("\nTo generate text from a trained model, run:")
    print("  python generate.py --checkpoint <path> --prompt 'Your prompt'")
    print("="*60)


if __name__ == '__main__':
    main()