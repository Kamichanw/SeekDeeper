from torch import nn

from .layers import *


class EncoderLayer(nn.Module):

    def __init__(self, hidden_size, ffn_hidden, num_attention_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
        )
        self.ln_1 = LayerNorm(normalized_shape=hidden_size)

        self.ffn = PositionwiseFeedForward(
            hidden_size=hidden_size, hidden=ffn_hidden, dropout=dropout
        )
        self.ln_2 = LayerNorm(normalized_shape=hidden_size)

    def forward(self, x, src_mask):
        # 1. Self-Attention sublayer
        # x: [batch_size, seq_len, hidden_size]
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


class Encoder(nn.Module):

    def __init__(
        self,
        hidden_size,
        ffn_hidden,
        num_attention_heads,
        num_hidden_layers,
        dropout,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_size=hidden_size,
                    ffn_hidden=ffn_hidden,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout,
                )
                for _ in range(num_hidden_layers)
            ]
        )

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)

        return x
