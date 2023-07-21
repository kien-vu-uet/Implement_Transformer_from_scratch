
from typing import Optional
import torch
import torch.nn as nn
from .encoder import TransformerEncoderBlock
from .embedding import VisionEmbedding

class VisionTransformer(nn.Module):
    def __init__(
            self,
            num_classes: int,
            clf_hidden_dim: int,
            img_size: int = 224,
            patch_size: int = 16,
            img_channels: int = 3,
            add_norm_to_embedding: bool = True,
            num_blocks: int = 6,
            embed_dim: int = 512,
            num_heads: int = 8,
            ff_dim: int = 2048,
            dropout: float = 0.1,
            qkv_bias: bool = False,
            attn_dropout: float = 0.1,
            ff_activate_fn: str = 'relu',
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
        super(VisionTransformer, self).__init__()
        self.embedding = VisionEmbedding(img_size, patch_size, img_channels, embed_dim, add_norm_to_embedding, dropout)
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout, qkv_bias, attn_dropout, ff_activate_fn)
                    for _ in range(num_blocks)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, clf_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(clf_hidden_dim, num_classes)
        )
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
            self,
            x: torch.Tensor,
            add_token_cls: bool = True,
            pooling_output: bool = True,
            apply_tanh: bool = False,
            output_attention: bool = False
    ):
        """
        Performs multi encoder blocks forward pass given input as token ids

        :param x: Input image. (B, C, H, W)
        :param output_attention: Return attention scores or not

        :return: Encoder hidden states. (B, S, E)

        B = batch size
        S = source sequence length
        E = embedding demensionality
        """
        x = self.embedding(x, add_token_cls=add_token_cls)
        for block in self.encoder_blocks:
            x, final_attn = block(x, output_attention=True)
        if apply_tanh:
            x = nn.Tanh()(x)
        if pooling_output:
            first_tokens_tensor = x[:, 0]
        x = self.classifier(first_tokens_tensor)
        if output_attention: return x, final_attn
        return x

if __name__ == "__main__":
    num_classes = 10

    x = torch.randn((2, 3, 224, 224))

    model = VisionTransformer(
        num_classes,
        clf_hidden_dim=1000,
    )
    model._reset_parameters()
    output, attn = model.forward(
                        x,
                        add_token_cls=True,
                        output_attention=True
                    )
    
    assert output.shape == (2, 10)
    assert torch.any(torch.isnan(output)) == False
    assert output.requires_grad == True
    assert attn.shape == (2, 8, 197, 197)
    assert torch.any(torch.isnan(attn)) == False
    assert attn.requires_grad == True
    # print(pooler)