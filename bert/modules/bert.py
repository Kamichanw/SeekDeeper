import os
import torch
import torch.nn as nn
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
        intermediate_size=3072, # 4*hidden_size
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
            # [batch_size, seq_len] -> [batch_size, 1, seq_len, seq_len]
            input_mask = input_mask.unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)


        x = self.embeddings(x, token_type_ids)
        x = self.encoder(x, input_mask)


        if self.pooler is not None:
            pooled_output = torch.tanh(self.pooler.dense(x[:, 0]))


            return x, pooled_output

        return x

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, num_frozen_layers):
        """
        Loads pretrained Bert model from transformers library.

        Args:
            model_name_or_path (str): Path to a directory or model name from HuggingFace hub.
        """

        import transformers

        try:
            # cls.__name__ is the name of the class, e.g. BertModel
            cls_name = "BertModel"
            HFModelClass = getattr(transformers, cls_name)
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

        # Check if there are any missing keys in our model
        if len(sd_keys_hf) > len(sd_keys):
            raise ValueError(
                "Some keys are missing in one of the models. "
                f"HF: {sd_keys_hf - sd_keys}, Ours: {sd_keys - sd_keys_hf}"
            )

        for k in sd_keys_hf:
            assert (
                sd_hf[k].shape == sd[k].shape
            ), f"Shape mismatch for key {k}: {sd_hf[k].shape} vs {sd[k].shape}"
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

        if num_frozen_layers > 0:
            layers_to_freeze = range(num_frozen_layers)
            for name, param in model.named_parameters():
                param.requires_grad = not any(f"encoder.layer.{i}." in name for i in layers_to_freeze)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params / 1e6:.2f}M")

        return model


# given hidden states of input sequence, output the prediction scores for each token in the vocab
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

        # weight tying: word_embeddings and vocab prediction decoder weights
        # word_embeddings (vocab_size, hidden_size); decoder.weight (hidden_size, vocab_size)
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

        prediction_scores = self.cls.predictions(sequence_output)
        seq_relationship_score = self.cls.seq_relationship(pooled_output)

        total_loss = None
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

class BERTTextClassifier(BertModel):
    def __init__(
        self,
        vocab_size,
        type_vocab_size=2,
        hidden_size=768,
        max_len=512,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size = 3072,
        dropout=0.1,
        add_pooling_layer=True,
        pad_token_idx=None,
        num_classes=2,
    ):
        # Initialize the parent BERT class with the given parameters
        super().__init__(
            vocab_size,
            type_vocab_size,
            hidden_size,
            max_len,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            dropout,
            add_pooling_layer,
            pad_token_idx
        )
        # classifier for text classification
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        x, pooled_output = super().forward(input_ids, token_type_ids, attention_mask)  # get the output from BERT
        logits = self.classifier(pooled_output)

        # if labels are provided, add the loss and return it
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits  # return the loss and logits for evaluation
        else:
            return logits  # return the logits for inference

    @torch.no_grad()
    # for inference
    def text_clf(self, input_ids):
        self.eval()
        logits = self.forward(input_ids)
        predicted_class = torch.argmax(logits, dim=1)
        predicted_class = predicted_class.cpu().numpy()
        return predicted_class