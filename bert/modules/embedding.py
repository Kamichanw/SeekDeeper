import torch
from torch import nn


class PositionalEmbedding(nn.Embedding):

    def __init__(self, d_model, max_len=512):  # default max_len=512
        super().__init__(max_len, d_model)

        # Initialize position encoding matrix (shape: [max_len, d_model])
        pe = torch.zeros(max_len, d_model)

        self.weight = nn.Parameter(pe)

    def forward(self, x):
        #  [1, seq_len, d_model]
        return self.weight[:x.size(1)].unsqueeze(0)


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        # consider only two segments A-0 and B-1 since pad 0 wouldn't be counted in due to mask mechanism
        super().__init__(2, embed_size, padding_idx=0)


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512,device=None, dtype=None):
        super().__init__(vocab_size, embed_size, padding_idx=0, device=device, dtype=dtype)
        self.weight = nn.Parameter(torch.empty((vocab_size, embed_size), dtype=dtype, device=device))
        self.reset_parameters()


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """
    def __init__(self, vocab_size, embed_size, dropout=0.1,device=None, dtype=None):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size, embed_size, device=device, dtype=dtype)
        self.position = PositionalEmbedding(embed_size)
        self.segment = SegmentEmbedding(embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label=None):
        if segment_label is None:
            # if segment_label is not provided, create a tensor of zeros with same shape as sequence
            segment_label = torch.zeros_like(sequence)
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)


