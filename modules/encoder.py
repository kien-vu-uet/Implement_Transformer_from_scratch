
import math
import torch
import torch.nn as nn
from typing import Optional
from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForwardLayer
from .embedding import TextEmbedding

class TransformerEncoder(nn.Module):
    def __init__(
            self,
            src_vocab_size: int,
            pad_token_id: Optional[int] = None,
            num_token_types: Optional[int] = None,
            num_blocks: int = 6,
            embed_dim: int = 512,
            num_heads: int = 8,
            ff_dim: int = 2048,
            dropout: float = 0.1,
            qkv_bias: bool = False,
            attn_dropout: float = 0.1,
            ff_activate_fn: str = 'relu'
    ):
        """
        :param src_vocab_size: The size of source's vocabulary
        :param pad_token_id: The index of pad token in source's vocabulary
        :param num_token_types: The number of token types use to embed when input is concatenated by multiple sequences
        :param num_blocks: Number of Encoder blocks
        :param embed_dim: Embedding dimensionality of input sequence
        :param num_heads: Number of heads in each block
        :param ff_dim: The feed forward dimensionality
        :param dropout: Dropout before feed forward
        :param qkv_bias: Use bias or not while projection query, key and value
        :param attn_dropout: Dropout when compute attention
        """
        super(TransformerEncoder, self).__init__()
        self.embedding = TextEmbedding(src_vocab_size, pad_token_id, num_token_types, embed_dim, dropout)
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout, qkv_bias, attn_dropout, ff_activate_fn)
                    for _ in range(num_blocks)
            ]
        )
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
            self,
            input_ids: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
            src_padding_mask: Optional[torch.BoolTensor] = None,
            output_attention: bool = True
    ):
        """
        Performs multi encoder blocks forward pass given input as token ids

        :param input_ids: Input token ids. (B, S)
        :param token_type_ids: The type of input tokens.
        :param src_padding_mask: Mask for handling pad tokens. (B, S)
        :param ff_activate_fn: Activate function for feed forward layer in each block
        :param output_attention: Return attention scores or not

        :return: Encoder hidden states. (B, S, E)

        B = batch size
        S = source sequence length
        E = embedding demensionality
        """
        x = self.embedding(input_ids, token_type_ids)
        for block in self.encoder_blocks:
            x, final_attn = block(x, src_padding_mask=src_padding_mask, output_attention=True)
        if output_attention: return x, final_attn
        return x


class TransformerEncoderBlock(nn.Module):
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
        :param ff_activate_fn: Activate function for feed forward layer in each block
        :param attn_dropout: Dropout when compute attention
        """
        super(TransformerEncoderBlock, self).__init__()
        self.self_mha = MultiHeadAttention(embed_dim, num_heads, qkv_bias, attn_dropout)
        self.feed_forward = FeedForwardLayer(embed_dim, ff_dim, dropout, activate_fn=ff_activate_fn)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
            self,
            x: torch.Tensor,
            src_padding_mask: Optional[torch.BoolTensor] = None,
            output_attention: Optional[bool] = True
    ):
        """
        Performs single encoder block forward pass given the previous block's output

        :param x: Source sequence. (B, S, E)
        :param src_padding_mask: Mask handling pad tokens. (B, S)

        :return: Updated encoded source sequence. (B, S, E)
        """
        mha_out, attn = self.self_mha(x, key_padding_mask=src_padding_mask, output_attention=True)
        x = self.norm1(self.dropout1(mha_out) + x)
        ff_out = self.feed_forward(x)
        output = self.norm2(self.dropout2(ff_out) + x)
        if output_attention: return output, attn
        return output


if __name__ == "__main__":
    input_ids = torch.randint(0, 10, (2, 5))
    src_padding_mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])
    # token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)

    encoder = TransformerEncoder(
        src_vocab_size=10,
        # pad_token_id=0,
        num_token_types=1,
        embed_dim=512,
        num_blocks=3,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1,
        qkv_bias=True,
        attn_dropout=0.1
    )

    encoder._reset_parameters()
    output, attn = encoder(
        input_ids, 
        src_padding_mask=src_padding_mask, 
        # token_type_ids=token_type_ids, 
        output_attention=True
        )
    assert output.shape == (2, 5, 512)
    assert torch.any(torch.isnan(output)) == False
    assert output.requires_grad == True
    assert attn.shape == (2, 8, 5, 5)
    assert torch.any(torch.isnan(attn)) == False