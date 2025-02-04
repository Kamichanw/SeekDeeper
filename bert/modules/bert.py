import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig

from .embedding import BERTEmbedding
from .transformer_encoder import TransformerBlock
from torch.nn import functional as F

class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, max_len=512, device=None, dtype=None):
        """
        BERT Base Model
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.max_len = max_len

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden,device=device, dtype=dtype)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, segment_info=None):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, num_frozen_layers=0,**bert_kwargs):
        """
        Loads pretrained BERT model from transformers library.

        Args:
            model_name_or_path (str): Path to a directory or model name from HuggingFace hub.
            num_frozen_layers (int): The number of layers whose parameters should be frozen.
            bert_kwargs: Additional keyword arguments for BERT initialization.
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32  # 或根据需求选择数据类型

        # 加载配置
        config = BertConfig.from_pretrained(model_name_or_path, **bert_kwargs)

        # Load the pre-trained model from HuggingFace using AutoModel
        bert_model = AutoModel.from_pretrained(model_name_or_path, config=config)
        bert_model = bert_model.to(device).to(dtype)

        # 初始化自定义 BERT 模型
        model = cls(
            vocab_size=config.vocab_size,
            hidden=config.hidden_size,
            n_layers=config.num_hidden_layers,
            attn_heads=config.num_attention_heads,
            dropout=config.hidden_dropout_prob,
            device=device,
            dtype=dtype
        )

        # Copy the pre-trained weights to our model
        model.embedding.token.weight.data.copy_(bert_model.embeddings.word_embeddings.weight.data)
        # position embedding is computed, do not need to copy (with sin/cos)
        # update with segment only 2 (0,1) to be consistent with the hf BERT
        model.embedding.segment.weight.data = bert_model.embeddings.token_type_embeddings.weight.data

        if num_frozen_layers > 0:
            layers_to_freeze = range(num_frozen_layers)
            for name, param in model.named_parameters():
                param.requires_grad = not any(f"encoder.layer.{i}." in name for i in layers_to_freeze)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params / 1e6:.2f}M")

        return model


class BERTTextClassifier(BERT):
    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, max_len=512, num_labels=2,device=None, dtype=None):
        # Initialize the parent BERT class with the given parameters
        super().__init__(vocab_size, hidden, n_layers, attn_heads, dropout, max_len, device, dtype)
        # classifier for text classification
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, segment_info=None, attention_mask=None, labels=None):
        x = super().forward(input_ids, segment_info)  # get the output from BERT
        cls_output = x[:, 0, :]  # get the CLS token output
        logits = self.classifier(cls_output)

        # if labels are provided, add the loss and return it
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits  # return the loss and logits for evaluation
        else:
            return logits  # return the logits for inference

    @torch.no_grad()
    def text_clf(self, input_ids):
        self.eval()

        logits = self.forward(input_ids)

        probs = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1)

        return predicted_class
