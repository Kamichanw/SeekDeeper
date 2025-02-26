import torch
from torch import nn

from .layers import *


class DecoderLayer(nn.Module):

    def __init__(self, hidden_size, ffn_hidden, num_attention_heads, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
        )
        self.ln_1 = LayerNorm(normalized_shape=hidden_size)

        self.enc_dec_attention = MultiheadAttention(
            hidden_size=hidden_size, num_attention_heads=num_attention_heads
        )
        self.ln_2 = LayerNorm(normalized_shape=hidden_size)

        self.ffn = PositionwiseFeedForward(
            hidden_size=hidden_size, hidden=ffn_hidden, dropout=dropout
        )
        self.ln_3 = LayerNorm(normalized_shape=hidden_size)

    def forward(self, dec, enc, tgt_mask, src_mask):
        # 1. Apply self attention
        residual = dec
        x = self.self_attn(q=dec, k=dec, v=dec, mask=tgt_mask)

        # 2. Add and norm
        x = self.ln_1(x + residual)

        if enc is not None:
            # 3. Cross attention
            residual = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)

            # 4. add and norm
            x = self.ln_2(x + residual)

        # 5. positionwise feed forward network
        residual = x
        x = self.ffn(x)

        # 6. add and norm
        x = self.ln_3(x + residual)
        return x


class Decoder(nn.Module):
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
                DecoderLayer(
                    hidden_size=hidden_size,
                    ffn_hidden=ffn_hidden,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout,
                )
                for _ in range(num_hidden_layers)
            ]
        )

    def forward(self, tgt, enc_src, tgt_mask, src_mask):
        for layer in self.layers:
            tgt = layer(tgt, enc_src, tgt_mask, src_mask)

        return tgt
