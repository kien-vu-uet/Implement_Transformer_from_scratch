import torch
import torch.nn as nn
from typing import List, Optional

class FeedForwardLayer(nn.Module):
    def __init__(
            self,
            embed_dim: int = 512,
            forward_dim: int = 2048,
            dropout: float = 0.1,
            activate_fn: str = "relu"
    ):
        """
        :param embed_dim: Embedding dimensionality of input sequence.
        :param forward_dim: Feed forward dimensionality.
        :param activate_fn: Activation function. Choose list ['relu', 'gelu', 'sigmoid', 'tanh']
        """
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(embed_dim, forward_dim)
        self.linear2 = nn.Linear(forward_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        if activate_fn == 'relu': self.activate_fn = nn.ReLU()
        elif activate_fn == 'gelu': self.activate_fn = nn.GELU()
        elif activate_fn == 'sigmoid': self.activate_fn = nn.Sigmoid()
        elif activate_fn == 'tanh': self.activate_fn = nn.Tanh()
        else: raise Exception(f"Activation {activate_fn} is not allowed!")
    
    def forward(
        self,
        x: torch.Tensor
    ):
        """
        Perform a feed forward layer pass given the output of multi-head attention block

        :param x: Attention score. (B, S or T, E)

        :return: Updated attention score. (B, S or T, E)

        B = batch size
        S = source sequence length
        T = target sequence length
        E = embedding dimensionality
        """
        output = self.linear2(self.dropout(self.activate_fn(self.linear1(x))))
        return output
    

class MLPBlock(nn.Sequential):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: List[int],
        add_norm: Optional[bool] = True,
        activate_fn: Optional[str] = 'relu',
        dropout: float=0.1
    ):
        """
        :param in_dim: Input dimensionality
        :param hidden_dim: List of the hidden layer dimensionality
        :param add_norm: Add LayerNorm or not
        :param activate_fn: Activation function. Choose list ['relu', 'gelu', 'sigmoid', 'tanh']
        :param dropout: The probability for the dropout layer.
        """
        if activate_fn == 'relu': activate_fn = nn.ReLU()
        elif activate_fn == 'gelu': activate_fn = nn.GELU()
        elif activate_fn == 'sigmoid': activate_fn = nn.Sigmoid()
        elif activate_fn == 'tanh': activate_fn = nn.Tanh()
        else: raise Exception(f"Activation {activate_fn} is not allowed!")
        layers = []
        for hidden in hidden_dim[:-1]:
            layers.append(nn.Linear(in_dim, hidden))
            if add_norm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(activate_fn)
            layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, hidden_dim[-1]))
        layers.append(nn.Dropout(dropout))
        super().__init__(*layers)
    

if __name__ == "__main__":
    ff = MLPBlock(512, [2048, 512], activate_fn='relu')
    x = torch.randn(2, 5, 512)
    output = ff(x)
    assert output.shape == (2, 5, 512)
    assert output.requires_grad == True
    assert torch.any(torch.isnan(output)) == False