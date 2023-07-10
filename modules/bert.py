
from typing import Optional
import torch
import torch.nn as nn
from .encoder import TransformerEncoder

class BERT(nn.Module):
    def __init__(
            self,
            num_classes: int,
            clf_hidden_dim: int,
            vocab_size: int,
            pad_token_id: Optional[int] = None,
            num_token_types: Optional[int] = None,
            num_encoder_blocks: int = 6,
            embed_dim: int = 512,
            num_heads: int = 8,
            ff_dim: int = 2048,
            dropout: float = 0.1,
            qkv_bias: bool = False,
            attn_dropout: float = 0.1,
            ff_activate_fn: str = 'relu',
    ):
        """
        :param num_classes: The size of classifier output
        :param clf_hidden_dim: The hidden dimensionality of classifier
        :param vocab_size: The size of vocabulary of source or target sequences 
        :param pad_token_id: The index of pad token in vocabulary
        :param num_encoder_blocks: Number of encoder or decoder blocks.  
        :param embed_dim: Embedding dimensionality of input sequence
        :param num_heads: Number of heads in each block
        :param ff_dim: The feed forward dimensionality
        :param dropout: Dropout before feed forward
        :param qkv_bias: Use bias or not while projection query, key and value
        :param attn_dropout: Dropout when compute attention
        :param ff_activate_fn: Activate function for feed forward layer in each block
        """
        super(BERT, self).__init__()
        self.encoder = TransformerEncoder(
                            vocab_size,
                            pad_token_id,
                            num_token_types,
                            num_encoder_blocks,
                            embed_dim,
                            num_heads,
                            ff_dim,
                            dropout,
                            qkv_bias,
                            attn_dropout,
                            ff_activate_fn
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
            input_ids: torch.Tensor,
            key_padding_mask: Optional[torch.BoolTensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            pooling_output: Optional[bool] = False,
            apply_tanh: Optional[bool] = False,
            output_attenttion: Optional[bool] = False
    ):
        """
        Performs transformer forward pass given input as token ids

        :param input_ids: Source token ids. (B, S)
        :param token_type_ids: The type of input source tokens. (B, S)
        :param key_padding_mask: Mask for handling pad tokens in source sequences. (B, S)
        :param output_attention: Return attention scores or not

        :return: Classifier states. (B, S, C)

        B = batch size
        S = source sequence length
        E = embedding demensionality
        C = number of classes
        """
        encoder_hidden_states, encoder_attn = self.encoder(
                                                    input_ids=input_ids,
                                                    token_type_ids=token_type_ids,
                                                    src_padding_mask=key_padding_mask,
                                                    output_attention=True
                                                )
        if apply_tanh: 
            encoder_hidden_states = nn.Tanh()(encoder_hidden_states)
        output = self.classifier(encoder_hidden_states)
        if pooling_output:
            output = output[:, 0]
        if output_attenttion: 
            return (
                output,
                encoder_attn
            )
        return output


if __name__ == "__main__":
    src_vocab_size = 100
    num_classes = 10

    src = torch.randint(0, src_vocab_size, (2, 5))
    src_padding_mask = torch.tensor([[True, True, True, True, True], [True, True, False, False, False]])

    model = BERT(num_classes, src_vocab_size, qkv_bias=True)
    model._reset_parameters()
    output, pooler, attn = model.forward(
                        src,
                        src_padding_mask,
                        output_attenttion=True
                    )
    
    assert output.shape == (2, 5, 512)
    assert torch.any(torch.isnan(output)) == False
    assert output.requires_grad == True
    assert pooler.shape == (2, num_classes)
    assert torch.any(torch.isnan(pooler)) == False
    assert pooler.requires_grad == True
    assert attn.shape == (2, 8, 5, 5)
    assert torch.any(torch.isnan(attn)) == False
    assert attn.requires_grad == True
    # print(pooler)