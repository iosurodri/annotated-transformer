import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.layers.aux_functions import clones

def dot_product_attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)  # Scaling factor for the output is sqrt(d_k) (avoids vanishing gradient)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # Compute similarity between query and key (and scale it)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # All penalized values' scores are set to -infty (-1e9 in practice)
    # Scores are normalized to (0, 1)
    attention_matrix = F.softmax(scores, dim=-1)
    if dropout is not None:
        attention_matrix = dropout(attention_matrix)
    attention_matrix = torch.matmul(attention_matrix, value)  #Filter the value vector with the attention matrix
    return attention_matrix, attention_matrix


class MultiHeadedAttention(nn.Module):

    attention_types = {
        'dot_product': dot_product_attention
    }

    def __init__(self, h, d_model, dropout=0.1, attention='dot_product'):
        """
        h: Number of attention heads
        d_model: Dimensionality of input vectors (must be divisible by h)
        Notice: In order for multihead attention to share a similar computational cost with
        single head attention, each head works with inputs of reduced dimensionality (d_model / h).
        """
        "Take in model size and number of heads."
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k (could be different)

        self.d_k = d_model // h  # Dimensionality of each attention head
        self.h = h
        # Learn projection matrices in order to reduce the dimensionality of the input for each head:
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # Linear layers will be applied to the whole dataset, before spliting it into each head
        self.attention = self.attention_types[attention]

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.shape[0]

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # x.view(nbatches, -1, self.h, self.d_k) divides each self.d_k dimensions into the different self.h heads:
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)