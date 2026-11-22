"""
GPT Model - Complete implementation.
This combines all components: embeddings, transformer blocks, and language modeling head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TransformerBlock
from .positional import LearnedPositionalEmbedding
import math


class GPT(nn.Module):
    """
    GPT Language Model.
    
    Architecture:
        1. Token embeddings + positional embeddings
        2. Stack of transformer blocks
        3. Final layer norm
        4. Language modeling head (projects to vocabulary)
    """
    
    def __init__(self, config):
        """
        Args:
            config: GPTConfig instance containing model hyperparameters
        """
        super().__init__()
        self.config = config
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Positional embedding layer
        self.position_embedding = LearnedPositionalEmbedding(
            config.n_positions, 
            config.n_embd
        )
        
        # Dropout for embeddings
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Language modeling head (maps to vocabulary logits)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying: share weights between token embeddings and lm_head
        # This is standard practice in language models
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        # Report number of parameters
        print(f"Number of parameters: {self.get_num_params()/1e6:.2f}M")
    
    def get_num_params(self, non_embedding=False):
        """
        Return the number of parameters in the model.
        
        Args:
            non_embedding: If True, subtract position embeddings
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.position_embedding.position_embeddings.weight.numel()
        return n_params
    
    def _init_weights(self, module):
        """
        Initialize weights following GPT-2 paper.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, idx, targets=None, use_cache=False, past_key_values=None):
        """
        Forward pass of GPT model.
        
        Args:
            idx: Input token indices (batch_size, seq_length)
            targets: Target token indices for computing loss (batch_size, seq_length)
            use_cache: Whether to return key-value cache for generation
            past_key_values: Cached key-values from previous forward pass
            
        Returns:
            If targets is provided: (logits, loss)
            If targets is None: logits
            If use_cache: (logits, present_key_values)
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.n_positions, \
            f"Sequence length {t} exceeds maximum position {self.config.n_positions}"
        
        # Get position indices
        if past_key_values is not None:
            # If using cache, positions continue from where we left off
            past_length = past_key_values[0][0].size(2)
            pos = torch.arange(past_length, past_length + t, dtype=torch.long, device=device)
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # Forward through embeddings
        tok_emb = self.token_embedding(idx)  # (batch, seq_len, n_embd)
        pos_emb = self.position_embedding(pos.unsqueeze(0))  # (1, seq_len, n_embd)
        x = self.drop(tok_emb + pos_emb)
        
        # Forward through transformer blocks
        present_key_values = [] if use_cache else None
        
        for i, block in enumerate(self.blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            if use_cache:
                x, present = block(x, use_cache=True, layer_past=layer_past)
                present_key_values.append(present)
            else:
                x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            # Flatten for cross entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1),
                ignore_index=-1
            )
        
        if use_cache:
            return logits, present_key_values
        
        if loss is not None:
            return logits, loss
        
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, 
                 use_cache=False):
        """
        Generate text autoregressively.
        
        Args:
            idx: Starting token indices (batch_size, seq_length)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
            top_p: If set, nucleus sampling (sample from top tokens with cumulative prob p)
            use_cache: Use KV cache for faster generation
            
        Returns:
            Generated token indices (batch_size, seq_length + max_new_tokens)
        """
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # Crop sequence if it exceeds max position
            idx_cond = idx if idx.size(1) <= self.config.n_positions else \
                       idx[:, -self.config.n_positions:]
            
            # Forward pass
            if use_cache and past_key_values is not None:
                # Only pass last token when using cache
                logits, past_key_values = self(
                    idx_cond[:, -1:], 
                    use_cache=True,
                    past_key_values=past_key_values
                )
            else:
                if use_cache:
                    logits, past_key_values = self(idx_cond, use_cache=True)
                else:
                    logits = self(idx_cond)
            
            # Get logits for last position
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Optional nucleus (top-p) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[..., 0] = False
                
                # Scatter back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('Inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        Estimate model flops utilization (MFU).
        
        Args:
            fwdbwd_per_iter: Number of forward-backward passes per iteration
            dt: Time per iteration in seconds
            
        Returns:
            MFU as a percentage
        """
        # Approximate number of FLOPs per token
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.n_positions
        
        # Approximation from PaLM paper
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        
        # Express throughput in model flops per second
        flops_achieved = flops_per_iter / dt
        
        # Hardware peak FLOPS (for A100 GPU: ~312 TFLOPS)
        flops_promised = 312e12  # Adjust based on your hardware
        
        mfu = flops_achieved / flops_promised
        return mfu
    
    @classmethod
    def from_pretrained(cls, model_type, override_config=None):
        """
        Load pretrained GPT-2 weights from HuggingFace.
        
        Args:
            model_type: One of 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
            override_config: Optional config overrides
            
        Returns:
            GPT model with pretrained weights
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        
        from transformers import GPT2LMHeadModel
        print(f"Loading pretrained {model_type} from HuggingFace...")
        
        # Load HuggingFace model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        # Create our model with matching config
        from ..training.config import get_gpt2_config
        config = get_gpt2_config(model_type)
        if override_config is not None:
            for k, v in override_config.items():
                setattr(config, k, v)
        
        model = GPT(config)
        sd = model.state_dict()
        
        # Copy weights
        # Note: This requires careful mapping between our names and HF names
        keys = sd_hf.keys()
        for k in keys:
            if k.endswith('.attn.bias') or k.endswith('.attn.masked_bias'):
                # These are buffers, not parameters
                continue
            # Direct copy for matching names
            if k in sd:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        return model