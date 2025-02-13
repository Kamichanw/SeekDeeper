from typing import Optional
import torch
from torch import nn
import math
import torch.nn.functional as F

from transformers import BertModel


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    See paper Figure 2 for details.
    """

    def __init__(
        self,
        vocab_size,
        type_vocab_size,
        hidden_size,
        max_len,
        dropout=0.1,
        pad_token_idx=None,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_idx
        )
        self.position_embeddings = nn.Embedding(max_len, hidden_size)

        # token type embeddings is used for Next Sentence Prediction (NSP) task
        # type_vocab_size typically equals 2
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = input_ids.size()

        # When its auto-generated, token_type_ids can be None
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)

        embeddings = (
            self.word_embeddings(input_ids)
            + self.token_type_embeddings(token_type_ids)
            + self.position_embeddings(
                torch.arange(seq_len, dtype=torch.long, device=input_ids.device).expand(
                    (1, -1)
                )
            )
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    """
    Standard self-attention operation, without any modifications
    """

    def __init__(self, hidden_size, num_attention_heads, dropout):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(dropout)

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def forward(self, x, mask=None):
        bsz, nh, nd = (
            x.size(0),
            self.num_attention_heads,
            self.attention_head_size,
        )

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(bsz, -1, nh, nd).transpose(1, 2)
        k = self.key(x).view(bsz, -1, nh, nd).transpose(1, 2)
        v = self.value(x).view(bsz, -1, nh, nd).transpose(1, 2)

        if self.flash:
            att = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
                attn_mask=mask,
                scale=None,  # scaled_dot_product_attention will scale automatically even scale is None
            )
        else:
            att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(nd)
            if mask is not None:
                att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = torch.matmul(att, v)  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # re-assemble all head outputs side by side
        y = att.transpose(1, 2).contiguous().view(bsz, -1, self.all_head_size)

        # output projection will be performed later
        return y


class BertLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        dropout,
    ):
        super().__init__()
        # I use a lot of `ModuleDict` here to simplify the code structure
        # while ensuring that the pre-trained weights of huggingface transformers can be loaded directly
        self.attention = nn.ModuleDict(
            dict(
                self=SelfAttention(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout,
                ),
                output=nn.ModuleDict(
                    dict(
                        dense=nn.Linear(hidden_size, hidden_size),
                        LayerNorm=nn.LayerNorm(hidden_size),
                    )
                ),
            )
        )
        self.output = nn.ModuleDict(
            dict(
                dense=nn.Linear(intermediate_size, hidden_size),
                LayerNorm=nn.LayerNorm(hidden_size),
            )
        )
        self.intermediate = nn.ModuleDict(
            dict(dense=nn.Linear(hidden_size, intermediate_size))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # in HF BertLayer implementation, the following operations are fused into one class
        attention = self.attention.self(x, mask)
        attention = self.dropout(self.attention.output.dense(attention))
        x = self.attention.output.LayerNorm(x + attention)

        # feed forward layer (BertIntermediate + BertOutput)
        intermediate_output = F.gelu(self.intermediate.dense(x))
        # second residual connection is done here
        layer_output = self.output.LayerNorm(
            x + self.dropout(self.output.dense(intermediate_output))
        )

        return layer_output


class BertEncoder(nn.Module):
    def __init__(
        self,
        num_hidden_layers,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        dropout,
    ):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                BertLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout,
                )
                for _ in range(num_hidden_layers)
            ]
        )

    def forward(self, x, mask=None):
        for layer in self.layer:
            x = layer(x, mask)
        return x
