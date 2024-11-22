import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, einops
import math

class Config():
    max_n: int = 128
    d_model: int = 32
    d_vocab: int = max_n + 3
    n_ctx: int = 4
    n_heads: int = 2

    @property
    def head_dim(self):
        return self.d_model // self.n_heads

assert Config.d_model % Config.n_heads == 0
cfg = Config()

class TokenEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        std = 1.0 / math.sqrt(cfg.d_model)
        self._weight = nn.Parameter(torch.randn((cfg.d_vocab, cfg.d_model)) * std)
        
    def forward(self, tokens):
        # Generate token embeddings     
        embeddings = self._weight[tokens, :]
        
        assert embeddings.shape == (*tokens.shape, cfg.d_model)        
        return embeddings

class PositionalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        std = 1.0 / math.sqrt(cfg.d_model)
        self._positions = nn.Parameter(torch.randn((cfg.n_ctx, cfg.d_model)) * std)
    
    def forward(self, tokens):
        # Generate positional embeddings
        position_embeddings = self._positions[:tokens.shape[-1], :].unsqueeze(0)
        return position_embeddings

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initialize with proper scaling
        std = 1.0 / math.sqrt(cfg.d_model)
        
        self._query_weight = nn.Parameter(torch.randn((cfg.n_heads, cfg.d_model, cfg.head_dim)) * std)
        self._query_bias = nn.Parameter(torch.zeros((cfg.n_heads, cfg.head_dim)), requires_grad=False)
        
        self._key_weight = nn.Parameter(torch.randn((cfg.n_heads, cfg.d_model, cfg.head_dim)) * std)
        self._key_bias = nn.Parameter(torch.zeros((cfg.n_heads, cfg.head_dim)), requires_grad=False)
        
        self._value_weight = nn.Parameter(torch.randn((cfg.n_heads, cfg.d_model, cfg.head_dim)) * std)
        self._value_bias = nn.Parameter(torch.zeros((cfg.n_heads, cfg.head_dim)), requires_grad=False)
        
        self._output_weight = nn.Parameter(torch.randn((cfg.n_heads, cfg.head_dim, cfg.d_model)) * std)
        self._output_bias = nn.Parameter(torch.zeros(cfg.d_model), requires_grad=False)  
        
        # self.register_buffer("_attention_mask_value", torch.tensor(-float("inf")))
        
    def forward(self, normalized_residual):
        # Query, Key, Value projections
        query = einsum(normalized_residual, self._query_weight, 
                      "batch q_pos d_model, n_heads d_model head_dim -> batch q_pos n_heads head_dim") + self._query_bias
        key = einsum(normalized_residual, self._key_weight, 
                    "batch k_pos d_model, n_heads d_model head_dim -> batch k_pos n_heads head_dim") + self._key_bias
        value = einsum(normalized_residual, self._value_weight, 
                      "batch k_pos d_model, n_heads d_model head_dim -> batch k_pos n_heads head_dim") + self._value_bias
        
        # Attention scores
        attention_scores = einsum(query, key, 
                                "batch q_pos n_heads head_dim, batch k_pos n_heads head_dim -> batch n_heads q_pos k_pos")
                
        # Scale and softmax
        self._attention_raw = attention_scores
        
        attention_probs = (attention_scores / math.sqrt(cfg.head_dim)).softmax(dim=-1)
        
        self._attention_pattern = attention_probs
        
        attention_output = einsum(attention_probs, value, 
                                "batch n_heads q_pos k_pos, batch k_pos n_heads head_dim -> batch q_pos n_heads head_dim")
        
        # Project back to d_model dimension
        output = einsum(attention_output, self._output_weight, 
                       "batch q_pos n_heads head_dim, n_heads head_dim d_model -> batch q_pos d_model") + self._output_bias
        
        self.OV = einsum(self._value_weight.detach().cpu(), self._output_weight.detach().cpu(),
                         "n_heads d_model_in head_dim, n_heads head_dim d_model_out -> n_heads d_model_in d_model_out")
        
        self.QK = einsum(self._query_weight.detach().cpu(), self._key_weight.detach().cpu(),
                         "n_heads d_model_q head_dim, n_heads d_model_k head_dim -> n_heads d_model_q d_model_k")
        
        self._attention_output = output
        
        return output
        

class AttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self._attention = Attention()

    def forward(self, residual_input):
        # Apply the attention mechanism to the input
        attention_output = self._attention(residual_input)
        
        # Add the residual connection (skip connection) to the attention output
        output = residual_input + attention_output

        # Detach the residual output from the computation graph and move it to the CPU for analysis
        self._residual_output = output.detach().cpu()
               
        return output  # Return the final output with the residual connection


class AttentionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self._token_embedding = TokenEmbedding()
        self._positional_embedding = PositionalEmbedding()
        self._transformer_block = AttentionLayer()
        
    def forward(self, tokens):
        # Combine token and positional embeddings
        hidden_states = self._token_embedding(tokens) + self._positional_embedding(tokens)
        
        # Pass through transformer block
        hidden_states = self._transformer_block(hidden_states)
        
        # Project back to vocabulary space
        logits = hidden_states @ self._token_embedding._weight.T
        
        # Store activations for analysis
        activations = dict()
        activations['token_embeddings'] = self._token_embedding._weight.detach().cpu()
        activations['positional_embeddings'] = self._positional_embedding._positions.detach().cpu()
        activations['attention_scores'] = self._transformer_block._attention._attention_raw.detach().cpu()
        activations['attention_pattern'] = self._transformer_block._attention._attention_pattern.detach().cpu()
        activations['attention_output'] = self._transformer_block._attention._attention_output.detach().cpu()
        activations['OV'] = self._transformer_block._attention.OV.detach().cpu()
        activations['QK'] = self._transformer_block._attention.QK.detach().cpu()
        
        return logits, activations