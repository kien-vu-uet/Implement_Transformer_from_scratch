import torch
import torch.nn as nn
from typing import Optional
from .positional_encoding import SinusoidEncoding
import math

class TextEmbedding(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        pad_token_id: Optional[int] = None,
        num_token_types: Optional[int] = None,
        embed_dim: int = 512,
        dropout: float = 0.1,
    ):
        """
        :param vocab_size: The size of source's vocabulary
        :param pad_token_id: The index of pad token in source's vocabulary
        :param num_token_types: The number of token types use to embed when input is concatenated by multiple sequences
        :param embed_dim: Embedding dimensionality of input sequence
        :param dropout: Dropout before feed forward
        """
        super(TextEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, pad_token_id)
        self.embed_token_type = False
        self.positional_encoding = SinusoidEncoding(embed_dim)
        if num_token_types is not None:
            self.embed_token_type = True
            self.token_type_embedding = nn.Embedding(num_token_types, embed_dim)
            self.register_buffer("token_type_ids", torch.zeros((1, 5000), dtype=torch.long), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ):
        """
        Perform embedding layer to transform text to sequence.

        :param input_ids: Input token ids. (B, S or T)
        :param token_type_ids: The type of input tokens.
        
        B = batch size
        S = source sequence length
        T = target sequence length
        E = embedding demensionality

        :return: Embedding sequence. (B, S or T, E)
        """
        x = self.embedding(input_ids) * math.sqrt(self.embed_dim)
        x = self.positional_encoding(x)
        if self.embed_token_type:
            if token_type_ids is None:
                token_type_ids = self.token_type_ids[:, :x.size(1)]
            x = x + self.token_type_embedding(token_type_ids)
        x = self.dropout(x)
        return x
    
class VisionEmbedding(nn.Module):
    def __init__(
        self,
        img_size: int=224,
        patch_size: int=16,
        img_channels: int=3,
        embed_dim: int=512,
        add_norm: bool = True,
        dropout: float=0.1,
    ):
        """
        :param img_size: The size of image.
        :param patch_size: The size of patch.
        :param img_channels: The number of channels in image.
        :param embed_dim: Embedding dimensionality of input sequence
        :param add_norm: Add LayerNorm or not
        :param dropout: Dropout before feed forward
        """
        super(VisionEmbedding, self).__init__()
        assert img_size % patch_size == 0, "Input shape indivisible by patch size!"
        self.img_channels = img_channels
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.seq_len = (img_size // patch_size) ** 2
        self.conv_proj = nn.Conv2d(img_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = None
        if add_norm:
            self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
            self, 
            x : torch.Tensor,
            add_cls_token: bool = False,
        ):
        """
        Perform vision embedding layer to transform image to sequence

        :param x: Input image. (B, C, H, W)
        :param add_cls_token: Add class token or not

        :return: Embedded sequence. (B, L, E)

        B = batch size
        C = number of iimage channels
        H = image height
        W = image weight
        L = sequence length
        E = embedding dimensionality
        """
        B, C, H, W = x.shape
        assert C == self.img_channels, f"Wrong image channels! Expected {self.img_channels} but got {C}!"
        assert H == W and H == self.img_size, f"Wrong image size! Expected {self.image_size} but got {H} and {W}!"
        x = self.conv_proj(x) # (B, E, nP, nP)
        x = x.flatten(-2, -1).permute(0, 2, 1) # (B, L, E)
        if add_cls_token:
            batch_cls_token = nn.Parameter(torch.zeros(B, 1, self.embed_dim, device=x.device))
            x = torch.cat([batch_cls_token, x], dim=1)
        if self.norm is not None:
            x = self.norm(x)
        x = self.dropout(x)
        return x

    
if __name__ == "__main__":
    input_ids = torch.randn((2, 3, 224, 224))
    src_padding_mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])
    # token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)

    encoder = VisionEmbedding(
        img_size=224,
        img_channels=3, 
        patch_size=16,
        embed_dim=512,
        add_norm=True,
        dropout=0.1,
    )

    encoder._reset_parameters()
    output = encoder(
        input_ids, 
        add_cls_token = True
        )
    assert output.shape == (2, 197, 512)
    assert torch.any(torch.isnan(output)) == False
    assert output.requires_grad == True