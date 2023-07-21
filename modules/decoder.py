
import math
import torch
import torch.nn as nn
from typing import Optional
from .multi_head_attention import MultiHeadAttention
from .positional_encoding import SinusoidEncoding
from .feed_forward import FeedForwardLayer
from .embedding import TextEmbedding

class TransformerDecoder(nn.Module):
    def __init__(
            self,
            tgt_vocab_size: int,
            pad_token_id: Optional[int] = None,
            num_token_types: Optional[int] = None,
            num_blocks: int = 6,
            embed_dim: int = 512,
            num_heads: int = 8,
            ff_dim: int = 2048,
            dropout: float = 0.1,
            qkv_bias: bool = False,
            attn_dropout: float = 0.1,
            ff_activate_fn: str = 'relu',
            tie_output_to_embedding: bool = True
    ):
        """
        :param tgt_vocab_size: The size of target's vocabulary
        :param pad_token_id: The index of pad token in target's vocabulary
        :param tgt_vocab_size: The size of vocabulary 
        :param num_blocks: Number of Decoder blocks
        :param embed_dim: Embedding dimensionality of input sequence
        :param num_heads: Number of heads in each block
        :param ff_dim: The feed forward dimensionality
        :param dropout: Dropout before feed forward
        :param qkv_bias: Use bias or not while projection query, key and value
        :param attn_dropout: Dropout when compute attention
        :param ff_activate_fn: Activate function for feed forward layer in each block
        :param tie_output_to_embedding: Use embedding weights to perform classifier output
        """
        super(TransformerDecoder, self).__init__()
        self.embedding = TextEmbedding(tgt_vocab_size, pad_token_id, num_token_types, embed_dim, dropout)
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(embed_dim, num_heads, ff_dim, dropout, qkv_bias, attn_dropout, ff_activate_fn)
                for _ in range(num_blocks)
            ]
        )
        self.classifier = nn.Linear(embed_dim, tgt_vocab_size)
        if tie_output_to_embedding: 
            self.classifier.weight = nn.Parameter(self.embedding.embedding.weight)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
            self,
            input_ids: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
            src_padding_mask: Optional[torch.BoolTensor] = None,
            tgt_padding_mask: Optional[torch.BoolTensor] = None,
            no_peeking_mask: Optional[torch.BoolTensor] = None, 
            output_attention: Optional[bool] = False
    ):
        """
        Performs multi decoder blocks forward pass given input as token ids

        :param input_ids: Input token ids. (B, T)
        :param encoder_hidden_states: Encoder hidden states to compute cross-attention with x. (B, S, E)
        :param token_type_ids: The type of input tokens.
        :param src_padding_mask: Mask for handling pad tokens in source sequence. (B, S)
        :param tgt_padding_mask: Mask for handling pad tokens in target sequence. (B, T)
        :param no_peeking_mask: Mask for avoiding any token i attending to a token >i. (T, T)
        :param output_attention: Return attention scores or not


        :return: Encoder hidden states. (B, S, E)

        B = batch size
        S = source sequence length
        E = embedding demensionality
        """
        x = self.embedding(input_ids, token_type_ids)
        for block in self.decoder_blocks:
            x, final_attn = block(
                x,
                encoder_hidden_states=encoder_hidden_states, 
                src_padding_mask=src_padding_mask, 
                tgt_padding_mask=tgt_padding_mask,
                no_peeking_mask = no_peeking_mask,
                output_attention=True)
        x = self.classifier(x)
        if output_attention: return x, final_attn
        return x
        


class TransformerDecoderBlock(nn.Module):
    def __init__(
            self,
            embed_dim: int = 512,
            num_heads: int = 8,
            ff_dim: int = 2048,
            dropout: float = 0.1,
            qkv_bias: bool = False,
            attn_dropout: float = 0.1,
            ff_activate_fn: str = 'relu'
    ):
        """
        :param embed_dim: Embedding dimensionality of input sequence
        :param num_heads: Number of heads in each block
        :param ff_dim: The feed forward dimensionality
        :param dropout: Dropout before feed forward
        :param qkv_bias: Use bias or not while projection query, key and value
        :param attn_dropout: Dropout when compute attention
        :param ff_activate_fn: Activate function for feed forward layer in each block
        """
        super(TransformerDecoderBlock, self).__init__()
        self.self_mha = MultiHeadAttention(embed_dim, num_heads, qkv_bias, attn_dropout)
        self.cross_mha = MultiHeadAttention(embed_dim, num_heads, qkv_bias, attn_dropout)
        self.feed_forward = FeedForwardLayer(embed_dim, ff_dim, dropout, ff_activate_fn)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(
            self, 
            x: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            src_padding_mask: Optional[torch.BoolTensor] = None,
            tgt_padding_mask: Optional[torch.BoolTensor] = None,
            no_peeking_mask: Optional[torch.BoolTensor] = None, 
            output_attention: Optional[bool] = False
    ):
        """
        Performs single encoder block forward pass given the previous block's output

        :param x: Target sequence. (B, T, E)
        :param encoder_hidden_states: Encoder hidden states to compute cross-attention with x. (B, S, E)
        :param src_padding_mask: Mask for handling pad tokens in source sequence. (B, S)
        :param tgt_padding_mask: Mask for handling pad tokens in target sequence. (B, T)
        :param no_peeking_mask: Mask for avoiding any token i attending to a token >i. (T, T)
        :param output_attention: Return attention scores or not

        :return: Updated decoder states. (B, T, E)
                 Attention scores if output_attention = True. (B, H, T, S)
        """
        smha = self.self_mha(x, key_padding_mask=tgt_padding_mask, no_peeking_mask=no_peeking_mask)
        x = self.norm1(self.dropout1(smha) + x)
        cmha, attn = self.cross_mha(x, 
                              encoder_hidden_states=encoder_hidden_states,
                              key_padding_mask=src_padding_mask, 
                              output_attention=True)
        x = self.norm2(self.dropout2(cmha) + x)
        ff_out = self.feed_forward(x)
        output = self.norm2(self.dropout2(ff_out) + x)
        if output_attention: return output, attn
        return output
    

if __name__ == "__main__":
    tgt_vocab_size = 100

    from utils import construct_no_peeking_mask
    input_ids = torch.randint(0, 10, (2, 5))
    token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
    tgt_padding_mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])
    no_peeking_mask = construct_no_peeking_mask(5)
    encoder_hidden_states = torch.randn(2, 5, 512, dtype=torch.float)
    src_padding_mask = torch.tensor([[True, True, True, True, True], [True, True, False, False, False]])
    

    decoder = TransformerDecoder(
        tgt_vocab_size=tgt_vocab_size,
        pad_token_id=0,
        num_token_types=1,
        embed_dim=512,
        num_blocks=3,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1,
        qkv_bias=True,
        attn_dropout=0.1, 
        tie_output_to_embedding=True
    )

    decoder._reset_parameters()
    output, attn = decoder(input_ids, 
                           encoder_hidden_states=encoder_hidden_states,
                           token_type_ids=token_type_ids,
                           src_padding_mask=src_padding_mask, 
                           tgt_padding_mask=tgt_padding_mask,
                           no_peeking_mask=no_peeking_mask,
                           output_attention=True)
    assert output.shape == (2, 5, tgt_vocab_size)
    assert torch.any(torch.isnan(output)) == False
    assert output.requires_grad == True
    assert attn.shape == (2, 8, 5, 5)
    assert torch.any(torch.isnan(attn)) == False