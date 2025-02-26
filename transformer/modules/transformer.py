import torch
from torch import nn

from .decoder import Decoder
from .encoder import Encoder
from .embedding import TransformerEmbedding


def make_pad_mask(seq, pad_idx):
    # [batch_size, 1, 1, src_len]
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask.to(seq.device)


def make_causal_mask(seq):
    batch_size, seq_len = seq.size()
    # [seq_len, seq_len]
    mask = torch.tril(torch.ones((seq_len, seq_len), device=seq.device)).bool()
    return mask


def make_tgt_mask(tgt, pad_idx):
    batch_size, tgt_len = tgt.shape
    # [batch_size, 1, 1, tgt_len]
    tgt_pad_mask = make_pad_mask(tgt, pad_idx)
    # [tgt_len, tgt_len]
    tgt_sub_mask = make_causal_mask(tgt)
    # [batch_size, 1, tgt_len, tgt_len]
    tgt_mask = tgt_pad_mask & tgt_sub_mask.unsqueeze(0).unsqueeze(0)
    return tgt_mask


class Transformer(nn.Module):

    def __init__(
        self,
        src_pad_idx,
        tgt_pad_idx,
        src_vocab_size,
        tgt_vocab_size,
        hidden_size,
        num_attention_heads,
        max_len,
        ffn_hidden,
        num_hidden_layers,
        dropout,
    ):
        """
        Constructor for the Transformer model.

        :param src_pad_idx: Padding index for the source sequences.
        :param tgt_pad_idx: Padding index for the target sequences.
        :param tgt_sos_idx: Start-of-sequence index for the target sequences.
        :param enc_voc_size: Vocabulary size of the encoder.
        :param dec_voc_size: Vocabulary size of the decoder.
        :param hidden_size: Dimensionality of the model.
        :param num_attention_heads: Number of attention heads.
        :param max_len: Maximum sequence length.
        :param ffn_hidden: Dimensionality of the feed-forward network.
        :param num_hidden_layers: Number of layers in the encoder and decoder.
        :param dropout: Dropout probability.
        """

        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.src_emb = TransformerEmbedding(
            hidden_size=hidden_size,
            max_len=max_len,
            vocab_size=src_vocab_size,
            dropout=dropout,
        )

        self.tgt_emb = TransformerEmbedding(
            hidden_size=hidden_size,
            dropout=dropout,
            max_len=max_len,
            vocab_size=tgt_vocab_size,
        )

        self.encoder = Encoder(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            ffn_hidden=ffn_hidden,
            dropout=dropout,
            num_hidden_layers=num_hidden_layers,
        )

        self.decoder = Decoder(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            ffn_hidden=ffn_hidden,
            dropout=dropout,
            num_hidden_layers=num_hidden_layers,
        )

        self.linear = nn.Linear(hidden_size, tgt_vocab_size, bias=False)

        self._reset_parameters()

    def forward(self, src, tgt):
        return self.decode(tgt, self.encode(src), self.make_src_mask(src))

    def make_src_mask(self, src):
        return make_pad_mask(src, self.src_pad_idx)

    def make_tgt_mask(self, tgt):
        return make_tgt_mask(tgt, self.tgt_pad_idx)

    def encode(self, src):
        return self.encoder(self.src_emb(src), self.make_src_mask(src))

    def decode(self, tgt, memory, memory_mask):
        return self.linear(
            self.decoder(
                self.tgt_emb(tgt), memory, self.make_tgt_mask(tgt), memory_mask
            )
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
