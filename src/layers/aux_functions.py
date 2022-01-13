import torch
import torch.nn as nn
import copy

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    "Mask out subsequent positions"
    # Useful only for sequence transduction tasks (guarantees auto-regressive property)
    attn_shape = (1, size, size)
    # The predictions for position i can depend only on the known outputs at positions less than i:
    subsequent_matrix = torch.logical_not(torch.triu(torch.ones(attn_shape, dtype=torch.bool), diagonal=1))
    return subsequent_matrix