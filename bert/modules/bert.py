import os
import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig

from torch.nn import functional as F

from .layers import *


def _init_weights(module):
    """Initialize the weights"""
    if isinstance(module, nn.Linear):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class BertModel(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(
        self,
        vocab_size,
        type_vocab_size=2,
        hidden_size=768,
        max_len=512,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        dropout=0.1,
        add_pooling_layer=True,
        pad_token_idx=None,
    ):

        super().__init__()

        # embedding for Bert, sum of positional, segment, token embeddings, see paper Figure 2
        self.embeddings = BertEmbeddings(
            vocab_size=vocab_size,
            type_vocab_size=type_vocab_size,
            hidden_size=hidden_size,
            max_len=max_len,
            dropout=dropout,
            pad_token_idx=pad_token_idx,
        )

        self.encoder = BertEncoder(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
        )

        self.pooler = (
            nn.ModuleDict(dict(dense=nn.Linear(hidden_size, hidden_size)))
            if add_pooling_layer
            else None
        )

        self.apply(_init_weights)

    def forward(self, x, token_type_ids=None, input_mask=None):
        if input_mask is not None and input_mask.dim() == 2:
            # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            input_mask = input_mask.unsqueeze(1).unsqueeze(2)

        x = self.embeddings(x, token_type_ids)
        x = self.encoder(x, input_mask)

        if self.pooler is not None:
            pooled_output = torch.tanh(self.pooler.dense(x[:, 0]))
            return x, pooled_output

        return x

    @classmethod
    def from_pretrained(cls, model_name_or_path: str):
        """
        Loads pretrained Bert model from transformers library.

        Args:
            model_name_or_path (str): Path to a directory or model name from HuggingFace hub.
        """

        import transformers

        try:
            HFModelClass = getattr(transformers, cls.__name__)
        except AttributeError:
            raise ValueError(f"Transformers library doesn't have a {cls_name} class")

        model_hf = HFModelClass.from_pretrained(model_name_or_path)
        sd_hf = model_hf.state_dict()
        config_hf: transformers.BertConfig = model_hf.config

        model = cls(
            vocab_size=config_hf.vocab_size,
            type_vocab_size=config_hf.type_vocab_size,
            hidden_size=config_hf.hidden_size,
            max_len=config_hf.max_position_embeddings,
            num_hidden_layers=config_hf.num_hidden_layers,
            num_attention_heads=config_hf.num_attention_heads,
            intermediate_size=config_hf.intermediate_size,
            dropout=config_hf.hidden_dropout_prob,
            pad_token_idx=config_hf.pad_token_id,
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


class BertLMPredictionHead(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.transform = nn.ModuleDict(
            dict(
                dense=nn.Linear(hidden_size, hidden_size),
                LayerNorm=nn.LayerNorm(hidden_size),
            )
        )

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # The following operations are fused into one class in HF BertOnlyMLMHead
        hidden_states = self.transform.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.transform.LayerNorm(hidden_states)

        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.predictions = BertLMPredictionHead(
            vocab_size=vocab_size, hidden_size=hidden_size
        )

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertForPreTraining(nn.Module):
    """
    Bert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.
    """
    
    def __init__(
        self,
        vocab_size,
        type_vocab_size=2,
        hidden_size=768,
        max_len=512,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        dropout=0.1,
        pad_token_idx=None,
    ):
        super().__init__()

        self.vocab_size = vocab_size

        self.bert = BertModel(
            vocab_size=vocab_size,
            type_vocab_size=type_vocab_size,
            hidden_size=hidden_size,
            max_len=max_len,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            dropout=dropout,
            pad_token_idx=pad_token_idx,
        )

        self.cls = nn.ModuleDict(
            dict(
                predictions=BertLMPredictionHead(
                    vocab_size=vocab_size, hidden_size=hidden_size
                ),
                seq_relationship=nn.Linear(hidden_size, 2),
            )
        )
        self.cls.apply(_init_weights)

        # weight tying
        self.cls.predictions.decoder.weight = (
            self.bert.embeddings.word_embeddings.weight
        )

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        next_sentence_label=None,
    ):
        sequence_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask
        )
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output
        )

        # if labels are provided, add the loss and return it
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.vocab_size), labels.view(-1)
            )
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            )
            total_loss = masked_lm_loss + next_sentence_loss
        
        output = (prediction_scores, seq_relationship_score)
        return ((total_loss,) + output) if total_loss is not None else output
