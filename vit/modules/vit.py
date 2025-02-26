import os
from typing import Union

import torch
import torch.nn as nn

from .layers import ViTEmbeddings, ViTEncoder


class ViTModel(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        hidden_size=768,
        num_channels=3,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        dropout=0.1,
    ):
        super().__init__()
        self.embeddings = ViTEmbeddings(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.encoder = ViTEncoder(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
        )
        self.layernorm = nn.LayerNorm(hidden_size)

        self.apply(self._init_weights)

    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)
        encoded_output = self.encoder(hidden_states)
        return self.layernorm(encoded_output)

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=0.02
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, ViTEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32), mean=0.0, std=0.02
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32), mean=0.0, std=0.02
            ).to(module.cls_token.dtype)


class ViTForImageClassification(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        hidden_size=768,
        num_channels=3,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        num_labels=10,
        dropout=0.1,
    ):
        super().__init__()
        self.vit = ViTModel(
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_channels=num_channels,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )
        self.classifier = nn.Linear(hidden_size, num_labels)

        # There are various ways to initialize classifier
        # In HF, classifier will be initialized to a truncated normal distribution with mean=0, stddev=0.02
        # In PyTorch, classifier will be initialized to a zero initialized linear layer.
        # In original paper, classifier should be a zero-initialized matrix without bias.
        # Anyway, we will replace it to be consistent with paper later, so it doesn’t matter how initialize it here.

        nn.init.normal_(self.classifier.weight, 0, 0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, pixel_values):
        hidden_states = self.vit(pixel_values)
        cls_output = hidden_states[:, 0]  # CLS token
        logits = self.classifier(cls_output)
        return logits

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, num_labels=10):

        from transformers import (
            ViTForImageClassification as HFViTForImageClassification,
        )

        # Load from HuggingFace pretrained weights
        model_hf = HFViTForImageClassification.from_pretrained(
            model_name_or_path, num_labels=num_labels, ignore_mismatched_sizes=True
        )
        sd_hf = model_hf.state_dict()
        config_hf = model_hf.config

        model = cls(
            num_labels=num_labels,
            image_size=config_hf.image_size,
            patch_size=config_hf.patch_size,
            hidden_size=config_hf.hidden_size,
            num_channels=config_hf.num_channels,
            num_hidden_layers=config_hf.num_hidden_layers,
            num_attention_heads=config_hf.num_attention_heads,
            intermediate_size=config_hf.intermediate_size,
            dropout=config_hf.hidden_dropout_prob,
        )
        sd = model.state_dict()

        # Ensure all the parameters align between HuggingFace model and our model
        sd_keys = set(sd.keys())
        sd_keys_hf = set(sd_hf.keys())

        if sd_keys != sd_keys_hf:
            raise ValueError(
                "Some keys are missing in one of the models. "
                f"HF: {sd_keys - sd_keys_hf}, Ours: {sd_keys_hf - sd_keys}"
            )

        for k in sd_keys:
            assert (
                sd_hf[k].shape == sd[k].shape
            ), f"Shape mismatch for key {k}: {sd_hf[k].shape} vs {sd[k].shape}"
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

        return model
