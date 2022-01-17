import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)  # Multiply by sqrt(self.d_model) for scaling outputs


class PositionalEncoding(nn.Module):
    """Implement the PE function
    PE(pos, 2i) = sin(pos/(10000**(2*i/d_model)))
    PE(pos, 2i+1) = cos(pos/(10000**(2*i/d_model)))
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)  # Is dropout really necessary in positional encoding?

        # Precompute the positional encodings up to "max_len", once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Positional encoding is implemented as a lookup table (all values are precomputed)
        # Only the x.shape[1] first precomputed terms are necessary
        # x = x + self.pe[:, :x.shape[1]]  # x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        x = x + self.pe[:x.shape[1], :]
        return self.dropout(x)