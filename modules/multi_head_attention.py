
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MultiHeadAttention(nn.Module):
    def __init__(
            self, 
            embed_dim: int = 512, 
            nheads: int = 8, 
            qkv_bias: bool = False, 
            attn_dropout: float = 0.1,
        ):
        """
        :param embed_dim: Embedding dimensionality of input sequence
        :param nheads: Number of heads in each block
        :param qkv_bias: Use bias or not while projection query, key and value
        :param attn_dropout: Dropout when compute attention
        """
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % nheads == 0, f'Embedding dimension ({embed_dim}) should be divisible by number of heads ({nheads})'
        
        self.embed_dim = embed_dim
        self.nheads = nheads
        self.head_dim = self.embed_dim // self.nheads
        self.qkv_bias = qkv_bias

        self.qkv_proj = nn.Linear(self.head_dim, 3 * self.head_dim, bias=qkv_bias)
        self.fc_out = nn.Linear(self.nheads * self.head_dim, self.embed_dim)
        self.dropout = nn.Dropout(attn_dropout)
        self._initialize_params()    

    def _initialize_params(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(
            self,
            x: torch.Tensor, 
            encoder_hidden_states: Optional[torch.Tensor] = None, 
            key_padding_mask: Optional[torch.BoolTensor] = None,
            no_peeking_mask: Optional[torch.BoolTensor] = None,
            output_attention: Optional[bool] = False
        ):
        """
        Compute attention with one projection matrix (May be faster than 3 projection for each input).

        B = batch size
        S = source sequence length
        T = target sequence length
        E = embedding dimensionality
        H = number of heads
        D = head dimensionality

        :param x: Either encoder or decoder hidden states. (B, S or T, E)
        :param encoder_hidden_states: Encoder hidden states to compute cross-attention with x. (B, S, E)
        :param key_padding_mask: Mask for handling pad tokens. (B, S or T)
        :param no_peeking_mask: Mask for avoiding any token i attending to a token >i. (T, T)
        :param output_attention: Return attention scores or not

        :return: Attention score. (B, S, E) for encoder self_attention and deoder cross-attention
                                  (B, T, E) for decoder self-attention
                 Query score. (B, H, S or T, S or T) if output_attention is True
        """
        if encoder_hidden_states is None:
            q, k, v = self._self_attention_projection(x)
        else:
            q, k, v = self._cross_attention_projection(encoder_hidden_states, x)
        logits = torch.einsum("bqhd,bkhd->bhqk", [q, k])
        if key_padding_mask is not None:
            logits = logits.masked_fill(
                key_padding_mask[:, None, None, :] == 0, float("-1e20")
            )
        if no_peeking_mask is not None:
            logits = logits.masked_fill(
                no_peeking_mask == 0, float("-1e20")
            )
        product = logits / math.sqrt(self.head_dim)
        scores = self.dropout(F.softmax(product, dim=-1))
        output = torch.einsum("bhql,blhd->bqhd", [scores, v]).flatten(-2, -1)
        output = self.fc_out(output)
        if output_attention: return output, scores
        return output


    def _self_attention_projection(self, x: torch.Tensor):
        """
        Project x (a.k.a query - q, key - k, value - v)

        :param x: Input. (B, S or T, E)
        
        :return: query - q, key - k, value - v. (B, S or T, H, D)
        """
        B, L, _ = x.size()
        x = x.reshape(B, L, self.nheads, self.head_dim)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        return q, k, v
        

    def _cross_attention_projection(
            self, 
            encoder_hidden_states: torch.Tensor,
            decoder_hidden_states: torch.Tensor
        ):
        """
        Project decoder hidden states into query (q) vectors and encoder hidden states into key (k) and value (v) vectors.
        
        :param encoder_hidden_states: Encoder hidden states. (B, S, E)
        :param decoder_hidden_states: Decoder hidden states. (B, T, E)
        
        :return q,: Query vectors. (B, T, H, D)
        :return k, v: Key vectors and Value vectors. (B, S, H, D)
        """
        B, S, _ = encoder_hidden_states.size()
        _, T, _ = decoder_hidden_states.size()
        encoder_hidden_states = encoder_hidden_states.reshape(B, S, self.nheads, self.head_dim)
        decoder_hidden_states = decoder_hidden_states.reshape(B, T, self.nheads, self.head_dim)
        weight_q, weight_kv = self.qkv_proj.weight.split([self.head_dim, 2 * self.head_dim])
        bias_q, bias_kv = None, None
        if self.qkv_bias:
            bias_q, bias_kv = self.qkv_proj.bias.split([self.head_dim, 2 * self.head_dim])
        q = F.linear(decoder_hidden_states, weight=weight_q, bias=bias_q)
        k, v = F.linear(encoder_hidden_states, weight=weight_kv, bias=bias_kv).chunk(2, dim=-1)
        return q, k, v
    
    
if __name__ == "__main__":
    from utils import construct_no_peeking_mask
    mha = MultiHeadAttention(512, 8, qkv_bias=True)
    x = torch.randn(2, 5, 512, dtype=torch.float)
    encoder_hidden_states = torch.randn(2, 5, 512, dtype=torch.float)
    key_padding_mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])
    # no_peeking_mask = construct_no_peeking_mask(5)
    output, attn = mha.forward(
        x, 
        encoder_hidden_states=encoder_hidden_states, 
        key_padding_mask=key_padding_mask, 
        output_attention=True
        )
    assert output.shape == (2, 5, 512)
    assert torch.any(torch.isnan(output)) == False
    assert attn.shape == (2, 8, 5, 5)
    assert torch.any(torch.isnan(attn)) == False



    
        