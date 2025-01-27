import torch.nn as nn

from .layers import *


class TransformerBlock(nn.Module):

    def __init__(self, d_model, n_head, ffn_hidden, dropout):
        super(TransformerBlock, self).__init__()
        self.self_attn = MultiheadAttention(
            d_model=d_model, n_head=n_head, dropout=dropout
        )
        self.ln_1 = LayerNorm(normalized_shape=d_model)

        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, dropout=dropout
        )
        self.ln_2 = LayerNorm(normalized_shape=d_model)

    def forward(self, x, src_mask):
        # 1. Self-Attention sublayer
        residual = x
        x = self.self_attn(x, x, x, src_mask)

        # 2. Add and norm
        x = self.ln_1(x + residual)

        # 3. Feed-Forward sublayer
        residual = x
        x = self.ffn(x)

        # 4. Add and norm
        x = self.ln_2(x + residual)

        return x
