from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ViTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, image_size, patch_size, num_channels, hidden_size):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, pixel_values):
        # pixel_values: [batch_size, num_channels, height, width]
        # after proj: [batch_size, hidden_size, height // patch_size, width // patch_size]
        # final: [batch_size, (height // patch_size) * (width // patch_size), hidden_size]
        return self.projection(pixel_values).flatten(2).transpose(1, 2)


class ViTEmbeddings(nn.Module):
    def __init__(self, image_size, patch_size, num_channels, hidden_size, dropout):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.patch_embeddings = ViTPatchEmbeddings(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            hidden_size=hidden_size,
        )
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches + 1, hidden_size)
        )
        self.dropout = nn.Dropout(dropout)

    def interpolate_pos_encoding(self, embeddings, height, width):
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.patch_embeddings.patch_size
        w0 = width // self.patch_embeddings.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1

        # shape: [1, num_patch, hidden_dim] -> [1, num_patch_per_row, num_patch_per_col, hidden_dim]
        patch_pos_embed = patch_pos_embed.reshape(
            1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim
        )

        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        assert (
            int(h0) == patch_pos_embed.shape[-2]
            and int(w0) == patch_pos_embed.shape[-1]
        )

        # shape: [1, hidden_dim, num_patch_per_row, num_patch_per_col] -> [1, num_patch_per_row, num_patch_per_col, hidden_dim]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # 2d interpolate to resize positional embeddings
        return self.dropout(
            embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        )


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

    def forward(self, x):
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
                scale=None,  # scaled_dot_product_attention will scale automatically even scale is None
            )
        else:
            att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(nd)
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = torch.matmul(att, v) # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # re-assemble all head outputs side by side
        y = att.transpose(1, 2).contiguous().view(bsz, -1, self.all_head_size)

        # output projection will be performed later
        return y


class ViTLayer(nn.Module):
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
                attention=SelfAttention(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout,
                ),
                output=nn.ModuleDict(dict(dense=nn.Linear(hidden_size, hidden_size))),
            )
        )
        self.layernorm_before = nn.LayerNorm(hidden_size)
        self.output = nn.ModuleDict(
            dict(dense=nn.Linear(intermediate_size, hidden_size))
        )
        self.intermediate = nn.ModuleDict(
            dict(dense=nn.Linear(hidden_size, intermediate_size))
        )
        self.layernorm_after = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # in HF ViT implementation, the following operations are fused into one class
        attention = self.attention.attention(self.layernorm_before(x))
        attention = self.dropout(self.attention.output.dense(attention))

        # first residual connection
        x = x + self.dropout(attention)

        # in ViT, layernorm is also applied after self-attention
        intermediate_output = F.gelu(self.intermediate.dense(self.layernorm_after(x)))

        # second residual connection is done here
        layer_output = x + self.dropout(self.output.dense(intermediate_output))
        return layer_output


class ViTEncoder(nn.Module):
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
                ViTLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout,
                )
                for _ in range(num_hidden_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x
