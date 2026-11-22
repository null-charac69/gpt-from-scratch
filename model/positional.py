"""
Positional Encoding for Transformer models.
GPT uses learned positional embeddings rather than sinusoidal ones.
"""

import torch
import torch.nn as nn
import math


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional embeddings.
    This is what GPT-2 uses - simple learned embeddings for each position.
    """
    
    def __init__(self, n_positions, n_embd):
        """
        Args:
            n_positions: Maximum sequence length
            n_embd: Embedding dimension
        """
        super().__init__()
        self.n_positions = n_positions
        self.n_embd = n_embd
        
        # Learnable position embeddings
        self.position_embeddings = nn.Embedding(n_positions, n_embd)
    
    def forward(self, position_ids=None, seq_length=None):
        """
        Args:
            position_ids: Optional position IDs (batch_size, seq_length)
            seq_length: Sequence length if position_ids not provided
            
        Returns:
            Position embeddings (batch_size, seq_length, n_embd) or (seq_length, n_embd)
        """
        if position_ids is None:
            # Create position IDs [0, 1, 2, ..., seq_length-1]
            position_ids = torch.arange(
                seq_length,
                dtype=torch.long,
                device=self.position_embeddings.weight.device
            )
            position_ids = position_ids.unsqueeze(0)  # (1, seq_length)
        
        return self.position_embeddings(position_ids)


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings as used in original Transformer paper.
    Not used in GPT, but included for completeness.
    """
    
    def __init__(self, n_positions, n_embd):
        """
        Args:
            n_positions: Maximum sequence length
            n_embd: Embedding dimension
        """
        super().__init__()
        self.n_positions = n_positions
        self.n_embd = n_embd
        
        # Create sinusoidal position embeddings
        pe = torch.zeros(n_positions, n_embd)
        position = torch.arange(0, n_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, n_embd, 2).float() * (-math.log(10000.0) / n_embd)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
    
    def forward(self, seq_length):
        """
        Args:
            seq_length: Length of sequence
            
        Returns:
            Position embeddings (seq_length, n_embd)
        """
        return self.pe[:seq_length, :]


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    Used in models like GPT-Neo, GPT-J, and LLaMA.
    More modern approach than learned embeddings.
    """
    
    def __init__(self, dim, max_seq_len=2048, base=10000):
        """
        Args:
            dim: Dimension (usually head_dim)
            max_seq_len: Maximum sequence length
            base: Base for frequency computation
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Compute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Build cache
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len):
        """Build cos/sin cache for given sequence length."""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos()[None, :, None, :])
        self.register_buffer('sin_cached', emb.sin()[None, :, None, :])
    
    def forward(self, seq_len):
        """
        Returns cos and sin for rotary embeddings.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Tuple of (cos, sin) tensors
        """
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        
        return (
            self.cos_cached[:, :seq_len, :, :],
            self.sin_cached[:, :seq_len, :, :]
        )
    
    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin):
        """
        Apply rotary position embeddings to queries and keys.
        
        Args:
            q: Queries (batch, seq_len, num_heads, head_dim)
            k: Keys (batch, seq_len, num_heads, head_dim)
            cos: Cosine values
            sin: Sine values
            
        Returns:
            Rotated q and k
        """
        # Split into first and second half
        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)
        
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        
        return q_embed, k_embed