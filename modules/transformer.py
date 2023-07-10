
from typing import Optional
import torch
import torch.nn as nn
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder

class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size: int,
            tgt_vocab_size: int,
            src_pad_token_id: Optional[int] = None,
            tgt_pad_token_id: Optional[int] = None,
            num_src_token_types: Optional[int] = None,
            num_tgt_token_types: Optional[int] = None,
            num_encoder_blocks: int = 6,
            num_decoder_blocks: int = 6,
            embed_dim: int = 512,
            num_heads: int = 8,
            ff_dim: int = 2048,
            dropout: float = 0.1,
            qkv_bias: bool = False,
            attn_dropout: float = 0.1,
            ff_activate_fn: str = 'relu',
            tie_output_to_embedding: bool = False
    ):
        """
        :param ###_vocab_size: The size of vocabulary of source or target sequences 
        :param ###_pad_token_id: The index of pad token in vocabulary
        :param num_######_blocks: Number of encoder or decoder blocks.  
        :param embed_dim: Embedding dimensionality of input sequence
        :param num_heads: Number of heads in each block
        :param ff_dim: The feed forward dimensionality
        :param dropout: Dropout before feed forward
        :param qkv_bias: Use bias or not while projection query, key and value
        :param attn_dropout: Dropout when compute attention
        :param tie_output_to_embedding: Use embedding weights to perform classifier output
        """
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(
                            src_vocab_size,
                            src_pad_token_id,
                            num_src_token_types,
                            num_encoder_blocks,
                            embed_dim,
                            num_heads,
                            ff_dim,
                            dropout,
                            qkv_bias,
                            attn_dropout,
                            ff_activate_fn
                        )
        self.decoder = TransformerDecoder(
                            tgt_vocab_size,
                            tgt_pad_token_id,
                            num_tgt_token_types,
                            num_decoder_blocks,
                            embed_dim,
                            num_heads,
                            ff_dim,
                            dropout,
                            qkv_bias,
                            attn_dropout,
                            ff_activate_fn,
                            tie_output_to_embedding
                        )
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            src_padding_mask: Optional[torch.BoolTensor] = None,
            tgt_padding_mask: Optional[torch.BoolTensor] = None,
            no_peeking_mask: Optional[torch.BoolTensor] = None,
            src_token_type_ids: Optional[torch.Tensor] = None,
            tgt_token_type_ids: Optional[torch.Tensor] = None,
            output_attenttion: Optional[bool] = False
    ):
        """
        Performs transformer forward pass given input as token ids

        :param src: Source token ids. (B, S)
        :param tgt: Target token ids. (B, T)
        :param src_token_type_ids: The type of input source tokens. (B, S)
        :param tgt_token_type_ids: The type of input target tokens. (B, T)
        :param src_padding_mask: Mask for handling pad tokens in source sequences. (B, S)
        :param tgt_padding_mask: Mask for handling pad tokens in target sequences. (B, T)
        :param no_peeking_mask: Mask for avoiding any token i attending to a token >i. (T, T)
        :param output_attention: Return attention scores or not

        :return: Classifier states. (B, T, V)

        B = batch size
        S = source sequence length
        T = target sequence length
        E = embedding demensionality
        V = vocabulary size
        """
        encoder_hidden_states, encoder_attn = self.encoder(
                                                    input_ids=src,
                                                    token_type_ids=src_token_type_ids,
                                                    src_padding_mask=src_padding_mask,
                                                    output_attention=True
                                                )
        decoder_output, decoder_attn = self.decoder(
                                                    input_ids=tgt,
                                                    encoder_hidden_states=encoder_hidden_states,
                                                    token_type_ids=tgt_token_type_ids,
                                                    src_padding_mask=src_padding_mask,
                                                    tgt_padding_mask=tgt_padding_mask,
                                                    no_peeking_mask=no_peeking_mask,
                                                    output_attention=True
                                                )
        if output_attenttion: 
            return (
                encoder_hidden_states,
                decoder_output,
                encoder_attn,
                decoder_attn
            )
        return (encoder_hidden_states, decoder_output)


if __name__ == "__main__":
    src_vocab_size = 100
    tgt_vocab_size = 150

    from utils import construct_no_peeking_mask
    src = torch.randint(0, src_vocab_size, (2, 5))
    tgt = torch.randint(0, tgt_vocab_size, (2, 6))
    src_padding_mask = torch.tensor([[True, True, True, True, True], [True, True, False, False, False]])
    tgt_padding_mask = torch.tensor([[True, True, True, False, False, False], [True, True, True, True, False, False]])
    no_peeking_mask = construct_no_peeking_mask(6)

    model = Transformer(src_vocab_size, tgt_vocab_size)
    model._reset_parameters()
    enc, output = model.forward(
                        src,
                        tgt,
                        src_padding_mask,
                        tgt_padding_mask,
                        no_peeking_mask,
                    )
    
    assert output.shape == (2, 6, tgt_vocab_size)
    assert torch.any(torch.isnan(output)) == False
    assert output.requires_grad == True
    # print(output)
    # assert attn.shape == (2, 8, 5, 5)
    # assert torch.any(torch.isnan(attn)) == False