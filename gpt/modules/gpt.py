import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F

from .layers import Block


class GPT(nn.Module):
    """GPT Language Model"""

    def __init__(
        self,
        vocab_size,
        max_len,
        hidden_size,
        num_attention_heads,
        num_hidden_layers,
        dropout,
    ):
        super().__init__()
        self.max_len = max_len
        self.transformer = nn.ModuleDict(
            dict(
                tokens_embed=nn.Embedding(vocab_size, hidden_size),
                positions_embed=nn.Embedding(max_len, hidden_size),
                drop=nn.Dropout(dropout),
                h=nn.ModuleList(
                    [
                        Block(
                            hidden_size=hidden_size,
                            num_attention_heads=num_attention_heads,
                            max_len=max_len,
                            dropout=dropout,
                        )
                        for _ in range(num_hidden_layers)
                    ]
                ),
            )
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        self.transformer.tokens_embed.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.positions_embed.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(
        self, lr, weight_decay, betas=(0.9, 0.999), device_type=None
    ):
        # start with all of the candidate parameters
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, num_frozen_layers=0, **gpt_kwargs
    ):
        """
        Loads pretrained from transformers library.

        Args:
            model_name_or_path (str):
                Can be "openai-gpt", "openai-community/openai-gpt", or a path to checkpoint directory,
                which is same as `PreTrainedModel.from_pretrained`.
            num_frozen_layers (int):
                The number of layers whose parameters need to be frozen. If it is not 0,
                then the `num_frozen_layers` GPT blocks close to the input will be frozen,
                that is, requires_grad=False.
            gpt_kwargs:
                The key-word args that passed to initialize GPT
        """
        from transformers import OpenAIGPTLMHeadModel

        default_config = dict(
            vocab_size=40478,
            max_len=512,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=12,
            dropout=0.1,
        )

        default_config.update(gpt_kwargs)

        model = cls(**default_config)
        # report number of parameters
        print("number of parameters: %.2fM" % (model.get_num_params() / 1e6,))

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = OpenAIGPTLMHeadModel.from_pretrained(model_name_or_path)
        model_hf.resize_token_embeddings(default_config["vocab_size"])
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        ignored_suffix = [
            ".attn.bias",  # buffer
            ".attn.masked_bias",  # mask
        ]
        sd_keys_hf = [
            k
            for k in sd_keys_hf
            if all(not k.endswith(suffix) for suffix in ignored_suffix)
        ]  # ignore these

        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        layers_to_frozen = range(num_frozen_layers)
        for name, param in model.named_parameters():
            param.requires_grad = not any(f"h.{i}." in name for i in layers_to_frozen)

        return model

    def forward(self, input, input_mask=None, return_hiden_states=False):
        device = input.device
        b, t = input.size()
        assert (
            t <= self.max_len
        ), f"Cannot forward sequence of length {t}, block size is only {self.max_len}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.tokens_embed(
            input
        )  # token embeddings of shape (b, t, hidden_size)
        pos_emb = self.transformer.positions_embed(
            pos
        )  # position embeddings of shape (t, hidden_size)
        x = self.transformer.drop(tok_emb + pos_emb)

        if input_mask is not None and input_mask.dim() == 2:
            # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            input_mask = input_mask.unsqueeze(1).unsqueeze(2)

        for block in self.transformer.h:
            x = block(x, input_mask)

        if return_hiden_states:
            return self.lm_head(x), x
        else:
            return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at max_len
            idx_cond = idx if idx.size(1) <= self.max_len else idx[:, -self.max_len :]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # only take the first output, i.e. lm_logits
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class GPTClassifier(GPT):
    def __init__(
        self,
        n_classes,
        vocab_size,
        max_len,
        hidden_size,
        num_attention_heads,
        num_hidden_layers,
        dropout,
    ):
        super().__init__(
            vocab_size,
            max_len,
            hidden_size,
            num_attention_heads,
            num_hidden_layers,
            dropout,
        )
        self.clf_head = nn.Linear(hidden_size, n_classes, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, input_mask=None):
        lm_logits, hidden_states = super().forward(
            input, input_mask=input_mask, return_hiden_states=True
        )
        clf_logits = self.clf_head(self.dropout(hidden_states[:, -1]))
        return lm_logits, clf_logits
