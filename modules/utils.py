import torch

def construct_no_peeking_mask(tgt_len: int):
    """
    Construct a binary mask that contains 1's for all valid connections and 0's for all outgoing future connections.

    :param tgt_len: The length of target sequence. 
    :return: (tgt_len, tgt_len) mask
    """
    subsequent_mask = torch.triu(torch.full((tgt_len, tgt_len), 1), diagonal=1).type(torch.BoolTensor)
    return subsequent_mask
