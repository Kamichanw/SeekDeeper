import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):

    def __init__(self, hidden_size, max_len):
        super(PositionalEncoding, self).__init__()

        # Initialize position encoding matrix (shape: [max_len, hidden_size])
        pe = torch.zeros(max_len, hidden_size)

        # Create a tensor of shape [max_len, 1] with position indices
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute the div_term (shape: [hidden_size//2]) for the sin and cos functions
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size)
        )

        # Apply sin/cos to even/odd indices in the position encoding matrix
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register pe as a buffer, not a parameter (no gradients needed)
        self.register_buffer("pe", pe)

    def forward(self, x):
        batch_size, seq_len = x.size()
        # [seq_len, hidden_size]
        return self.pe[:seq_len, :]


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_len, dropout):
        super(TransformerEmbedding, self).__init__()

        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = PositionalEncoding(hidden_size, max_len)
        self.drop_out = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: [batch_size, seq_len]
        tok_emb = self.tok_emb(x)  # [batch_size, seq_len, hidden_size]
        pos_emb = self.pos_emb(x)  # [seq_len, hidden_size]

        # [batch_size, seq_len, hidden_size]
        return self.drop_out(tok_emb + pos_emb)
