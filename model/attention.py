"""
Multi-Head Self-Attention mechanism for GPT.
This is the core component that allows the model to attend to different positions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with a causal mask.
    Causal means the model can only attend to previous tokens, not future ones.
    This is what makes GPT autoregressive.
    """
    
    def __init__(self, config):
        """
        Args:
            config: GPTConfig instance
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        
        # Key, Query, Value projections for all heads (in a batch)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask to ensure attention is only applied to the left in the sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_positions, config.n_positions)).view(
                1, 1, config.n_positions, config.n_positions
            )
        )
    
    def forward(self, x, use_cache=False, layer_past=None):
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor (batch_size, seq_length, n_embd)
            use_cache: Whether to return past key-values for generation
            layer_past: Cached key-values from previous forward pass
            
        Returns:
            Output tensor (batch_size, seq_length, n_embd)
            Optionally, present key-values if use_cache=True
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality
        
        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape to (batch, num_heads, seq_length, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Handle past key-values for efficient generation
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)
        
        present = (k, v) if use_cache else None
        
        # Fast path: Scaled Dot-Product Attention (PyTorch 2.0+ FlashAttention/cuDNN)
        if layer_past is None and hasattr(F, 'scaled_dot_product_attention'):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # Manual causal self-attention
            # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            
            # Apply causal mask
            seq_len = k.size(2)
            att = att.masked_fill(self.bias[:, :, :T, :seq_len] == 0, float('-inf'))
            
            # Softmax and dropout
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            
            # Apply attention to values
            # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = att @ v
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        
        if use_cache:
            return y, present
        return y


class FlashAttention(nn.Module):
    """
    Simplified Flash Attention implementation.
    More memory efficient for long sequences.
    In practice, you'd use torch.nn.functional.scaled_dot_product_attention
    """
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        
        # QKV projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_positions, config.n_positions)).view(
                1, 1, config.n_positions, config.n_positions
            )
        )
    
    def forward(self, x):
        """
        Forward pass using PyTorch's optimized attention.
        
        Args:
            x: Input tensor (batch_size, seq_length, n_embd)
            
        Returns:
            Output tensor (batch_size, seq_length, n_embd)
        """
        B, T, C = x.size()
        
        # QKV projection
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Create causal mask
        causal_mask = self.bias[:, :, :T, :T]
        
        # Use PyTorch's scaled_dot_product_attention (Flash Attention when available)
        if hasattr(F, 'scaled_dot_product_attention'):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # Fallback to manual implementation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(causal_mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            if self.training:
                att = F.dropout(att, p=self.dropout)
            y = att @ v
        
        # Reassemble heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        
        return y


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA) - used in some newer models.
    Uses single key/value head shared across all query heads.
    More efficient for inference.
    """
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        
        # Query for all heads, but single key and value
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, self.head_dim)
        self.v_proj = nn.Linear(config.n_embd, self.head_dim)
        
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_positions, config.n_positions)).view(
                1, 1, config.n_positions, config.n_positions
            )
        )
    
    def forward(self, x):
        """
        Forward pass of multi-query attention.
        
        Args:
            x: Input tensor (batch_size, seq_length, n_embd)
            
        Returns:
            Output tensor (batch_size, seq_length, n_embd)
        """
        B, T, C = x.size()
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape Q for multi-head, K and V stay single-headed
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, 1, self.head_dim).transpose(1, 2)
        v = v.view(B, T, 1, self.head_dim).transpose(1, 2)
        
        # Expand K and V to match Q's head dimension
        k = k.expand(-1, self.n_head, -1, -1)
        v = v.expand(-1, self.n_head, -1, -1)
        
        # Attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to values
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        
        return y