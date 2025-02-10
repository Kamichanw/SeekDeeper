import torch
from torch import nn
import math
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            # Learnable parameters
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x):
        # x: [batch_size, ..., normalized_shape]
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x_normalized = (x - mean) / (std + self.eps)

        if self.elementwise_affine:
            x_normalized = self.gamma * x_normalized + self.beta

        return x_normalized


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.flash = self.flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, q, k, v, mask=None):
        if self.flash:
            out = F.scaled_dot_product_attention(q, k, v, mask)
        else:
            # calculate attention manually
            batch_size, n_head, seq_len, d_key = k.size()

            # 1. Compute the dot product between query and key^T
            k_t = k.transpose(-2, -1)
            scores = q @ k_t / math.sqrt(d_key)

            # 2. Apply mask (optional)
            if mask is not None:
                scores = scores.masked_fill(mask.logical_not(), float("-inf"))

            # 3. Apply softmax to get attention weights
            attn = F.softmax(scores, dim=-1)

            # 4. Compute the weighted sum of values
            out = attn @ v

        return out


class MultiheadAttention(nn.Module):

    def __init__(self, d_model, n_head, dropout):
        super(MultiheadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaledDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        # 1. Linear projections, [batch_size, length, d_model]
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. Split tensor by number of heads, [batch_size, length, n_head, d_key]
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. Apply attention
        out = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer, [batch_size, length, d_model]
        out = self.concat(out)
        out = self.w_concat(out)

        return self.dropout(out)

    def split(self, tensor):
        batch_size, length, d_model = tensor.size()

        d_key = d_model // self.n_head
        return tensor.view(batch_size, length, self.n_head, d_key).transpose(1, 2)

    def concat(self, tensor):
        batch_size, head, length, d_key = tensor.size()
        d_model = head * d_key

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class GELU(nn.Module):
    """
    Appendix A.2 notice that BERT used the GELU instead of RELU
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    """
    FFN layer
    """
    def __init__(self, d_model, hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):  # only encoder block

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
