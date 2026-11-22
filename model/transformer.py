"""
Transformer Block - the building block of GPT.
Each block contains attention and feed-forward layers with residual connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import CausalSelfAttention


class MLP(nn.Module):
    """
    Feed-Forward Network (MLP) used in transformer blocks.
    Simple two-layer network with GELU activation.
    """
    
    def __init__(self, config):
        """
        Args:
            config: GPTConfig instance
        """
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()  # GPT-2 uses GELU
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_length, n_embd)
            
        Returns:
            Output tensor (batch_size, seq_length, n_embd)
        """
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    A single Transformer block.
    
    Architecture:
        x -> LayerNorm -> Attention -> + (residual) 
          -> LayerNorm -> MLP -> + (residual) -> output
    """
    
    def __init__(self, config):
        """
        Args:
            config: GPTConfig instance
        """
        super().__init__()
        
        # Layer normalization
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Attention
        self.attn = CausalSelfAttention(config)
        
        # Feed-forward network
        self.mlp = MLP(config)
    
    def forward(self, x, use_cache=False, layer_past=None):
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor (batch_size, seq_length, n_embd)
            use_cache: Whether to use KV cache for generation
            layer_past: Past key-values if using cache
            
        Returns:
            Output tensor and optionally present key-values
        """
        # Attention with residual connection
        if use_cache:
            attn_output, present = self.attn(self.ln_1(x), use_cache=True, layer_past=layer_past)
            x = x + attn_output
        else:
            x = x + self.attn(self.ln_1(x))
            present = None
        
        # Feed-forward with residual connection
        x = x + self.mlp(self.ln_2(x))
        
        if use_cache:
            return x, present
        return x


class ParallelTransformerBlock(nn.Module):
    """
    Parallel Transformer Block (used in GPT-J, GPT-NeoX).
    Applies attention and MLP in parallel instead of sequentially.
    Slightly faster but different from original GPT-2.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
    
    def forward(self, x):
        """
        Forward pass with parallel attention and MLP.
        
        Args:
            x: Input tensor (batch_size, seq_length, n_embd)
            
        Returns:
            Output tensor (batch_size, seq_length, n_embd)
        """
        # Single layer norm, then parallel attention and MLP
        h = self.ln_1(x)
        attn_out = self.attn(h)
        mlp_out = self.mlp(h)
        
        # Combine and add residual
        x = x + attn_out + mlp_out
        return x


class PreNormTransformerBlock(nn.Module):
    """
    Pre-normalization transformer block (standard GPT-2 style).
    Layer norm is applied before each sub-layer.
    """
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class PostNormTransformerBlock(nn.Module):
    """
    Post-normalization transformer block (original Transformer paper style).
    Layer norm is applied after each sub-layer.
    Less stable training than pre-norm but included for completeness.
    """
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = self.ln_1(x + self.attn(x))
        x = self.ln_2(x + self.mlp(x))
        return x


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (used in LLaMA and newer models).
    Simpler and faster than LayerNorm.
    """
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        x = x / rms
        return self.weight * x


class TransformerBlockWithRMSNorm(nn.Module):
    """
    Transformer block using RMSNorm instead of LayerNorm.
    Used in modern models like LLaMA.
    """
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x