import math
import torch
import torch.nn as nn


class SinusoidEncoding(nn.Module):
    def __init__(
            self,
            embed_dim: int = 512,
            max_length: int = 5000,
            add_to_state_dict: bool = False
    ):
        super(SinusoidEncoding, self).__init__()
        positional_encoding = torch.zeros(max_length, embed_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
                        torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.) / embed_dim)
        )
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        """
        torch.register_buffer make Tensor should be part of the modules state, but not a parameter
            :param persitent: add the buffer to the state dict or not
        """
        self.register_buffer('positional_encoding', positional_encoding, persistent=add_to_state_dict)

    def forward(
            self,
            x: torch.Tensor
    ):
        """
        Perform and add positional encoding to embedded sequence

        B = batch size
        L = sequence length
        E = embedding dimensionality

        :param x: Input sequence which are embedded. (B, L, E)
        
        :return: Embedded sequence with positional encoding. (B, L, E)
        """
        x = x + self.positional_encoding[:, : x.size(1)]
        return x
    

if __name__ == "__main__":
    batch = 2
    dim = 8
    len = 3
    x = torch.zeros(batch, len, dim)
    encoding = SinusoidEncoding(dim).forward(x)
    expected = torch.Tensor(
        [
            [
                [
                    0.0000e00,
                    1.0000e00,
                    0.0000e00,
                    1.0000e00,
                    0.0000e00,
                    1.0000e00,
                    0.0000e00,
                    1.0000e00,
                ],
                [
                    8.4147e-01,
                    5.4030e-01,
                    9.9833e-02,
                    9.9500e-01,
                    9.9998e-03,
                    9.9995e-01,
                    1.0000e-03,
                    1.0000e00,
                ],
                [
                    9.0930e-01,
                    -4.1615e-01,
                    1.9867e-01,
                    9.8007e-01,
                    1.9999e-02,
                    9.9980e-01,
                    2.0000e-03,
                    1.0000e00,
                ],
            ],
            [
                [
                    0.0000e00,
                    1.0000e00,
                    0.0000e00,
                    1.0000e00,
                    0.0000e00,
                    1.0000e00,
                    0.0000e00,
                    1.0000e00,
                ],
                [
                    8.4147e-01,
                    5.4030e-01,
                    9.9833e-02,
                    9.9500e-01,
                    9.9998e-03,
                    9.9995e-01,
                    1.0000e-03,
                    1.0000e00,
                ],
                [
                    9.0930e-01,
                    -4.1615e-01,
                    1.9867e-01,
                    9.8007e-01,
                    1.9999e-02,
                    9.9980e-01,
                    2.0000e-03,
                    1.0000e00,
                ],
            ],
        ]
    )
    torch.testing.assert_close(encoding, expected, rtol=10e-5, atol=10e-5)
