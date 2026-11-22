"""
Text generation script for GPT.
Load a trained model and generate text.

Usage:
    python generate.py --checkpoint runs/run_XXXXXX/final_model.pt --prompt "Once upon a time"
"""

import torch
import argparse
from model.gpt import GPT
from training.config import GPTConfig
from data.tokenizer import SimpleTokenizer
from utils.helpers import get_device, set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate text with GPT')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default='',
                       help='Starting prompt for generation')
    parser.add_argument('--max_tokens', type=int, default=100,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=None,
                       help='Top-k sampling (sample from top k tokens)')
    parser.add_argument('--top_p', type=float, default=None,
                       help='Nucleus sampling (sample from top p probability mass)')
    parser.add_argument('--num_samples', type=int, default=1,
                       help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--use_cache', action='store_true',
                       help='Use KV cache for faster generation')
    
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Tuple of (model, config, tokenizer)
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct config
    config_dict = checkpoint['model_config']
    config = GPTConfig(**config_dict)
    
    # Reconstruct tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.chars = checkpoint['tokenizer_chars']
    tokenizer.vocab_size = len(tokenizer.chars)
    tokenizer.char_to_idx = {ch: i for i, ch in enumerate(tokenizer.chars)}
    tokenizer.idx_to_char = {i: ch for i, ch in enumerate(tokenizer.chars)}
    
    # Create and load model
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Context length: {config.n_positions}")
    
    return model, config, tokenizer


def generate_text(model, tokenizer, prompt, max_tokens, temperature, top_k, top_p, 
                 device, use_cache=False):
    """
    Generate text from a prompt.
    
    Args:
        model: GPT model
        tokenizer: Tokenizer
        prompt: Starting text
        max_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        device: Device to run on
        use_cache: Use KV cache for generation
        
    Returns:
        Generated text
    """
    # Encode prompt
    if prompt:
        context = torch.tensor(
            tokenizer.encode(prompt),
            dtype=torch.long,
            device=device
        ).unsqueeze(0)
    else:
        # Start with a random token if no prompt
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_cache=use_cache
        )
    
    # Decode
    generated_text = tokenizer.decode(generated[0].tolist())
    
    return generated_text


def interactive_mode(model, tokenizer, config, device, args):
    """
    Interactive generation mode.
    
    Args:
        model: GPT model
        tokenizer: Tokenizer
        config: Model config
        device: Device
        args: Command line arguments
    """
    print("\n" + "="*50)
    print("INTERACTIVE MODE")
    print("="*50)
    print("Enter a prompt and the model will continue the text.")
    print("Type 'quit' to exit.")
    print("="*50 + "\n")
    
    while True:
        # Get prompt
        prompt = input("Prompt (or 'quit'): ").strip()
        
        if prompt.lower() == 'quit':
            break
        
        # Generate
        print("\nGenerating...\n")
        generated = generate_text(
            model, tokenizer, prompt,
            args.max_tokens, args.temperature,
            args.top_k, args.top_p, device,
            args.use_cache
        )
        
        print("-" * 50)
        print(generated)
        print("-" * 50 + "\n")


def main():
    """Main generation function."""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    model, config, tokenizer = load_model(args.checkpoint, device)
    
    # Print generation settings
    print("\n" + "="*50)
    print("GENERATION SETTINGS")
    print("="*50)
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}")
    print(f"Top-p: {args.top_p}")
    print(f"Use cache: {args.use_cache}")
    print("="*50 + "\n")
    
    # Generate samples
    if args.prompt or args.num_samples == 1:
        # Single generation with prompt
        for i in range(args.num_samples):
            if args.num_samples > 1:
                print(f"\n{'='*50}")
                print(f"SAMPLE {i+1}/{args.num_samples}")
                print(f"{'='*50}\n")
            
            generated = generate_text(
                model, tokenizer, args.prompt,
                args.max_tokens, args.temperature,
                args.top_k, args.top_p, device,
                args.use_cache
            )
            
            print(generated)
            print()
    else:
        # Interactive mode
        interactive_mode(model, tokenizer, config, device, args)


if __name__ == '__main__':
    main()