from typing import Sequence, Union

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

from modules import make_pad_mask
import data
import config


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, ignore_index, label_smoothing):
        super(LabelSmoothingLoss, self).__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, pred, target):
        smoothing = self.label_smoothing
        pad_idx = self.ignore_index
        classes = pred.shape[-1]

        if smoothing == 0:
            return F.cross_entropy(pred, target, ignore_index=pad_idx)

        log_probs = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
            one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (
                classes - 1
            )
            mask = torch.nonzero(target == pad_idx)
            if mask.dim() > 0:
                one_hot.index_fill_(0, mask.squeeze(), 0.0)
        return torch.mean(torch.sum(-one_hot * log_probs, dim=-1))


class WarmupScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        hidden_size: int,
        warmup_step: int,
        last_epoch: int = -1,
    ) -> None:
        self.hidden_size = hidden_size
        self.warmup_step = warmup_step
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.hidden_size**-0.5
            * min(
                (self.last_epoch + 1) ** -0.5,
                (self.last_epoch + 1) * self.warmup_step**-1.5,
            )
        ]


@torch.no_grad
def greedy_search(model, memory, memory_mask, max_len, sos_idx, eos_idx, pad_idx):
    batch_size, seq_len, hidden_size = memory.shape
    ys = torch.ones(batch_size, 1, dtype=torch.long, device=memory.device).fill_(
        sos_idx
    )
    ended = torch.zeros(batch_size, dtype=torch.bool, device=memory.device)

    for i in range(max_len - 1):
        logits = model.decode(ys, memory, memory_mask)[:, -1]
        next_words = torch.argmax(logits, dim=1)

        ys = torch.cat([ys, next_words.unsqueeze(1)], dim=1)
        ended = ended | (next_words == eos_idx)
        ys[ended & (ys[:, -1] != eos_idx), -1] = pad_idx

        if ended.all():
            break

    if i == max_len - 2:  # reach max length
        ys[~ended, -1] = eos_idx
        ys[ended, -1] = pad_idx

    return ys


@torch.no_grad
def sample(
    model,
    memory,
    memory_mask,
    temperature,
    top_k,
    top_p,
    max_len,
    sos_idx,
    eos_idx,
    pad_idx,
):
    device = memory.device
    batch_size, seq_len, hidden_size = memory.shape
    ys = torch.ones(batch_size, 1, dtype=torch.long, device=device).fill_(sos_idx)
    ended = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for i in range(max_len - 1):
        logits = model.decode(ys, memory, memory_mask)[:, -1]
        logits = logits / temperature

        # Top-k sampling
        if top_k > 0:
            top_k_values, _ = torch.topk(logits, top_k)
            min_top_k_values = top_k_values[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_top_k_values,
                torch.full_like(logits, float("-inf")),
                logits,
            )

        # Top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            indices_to_remove = cumulative_probs > top_p
            indices_to_remove[:, 1:] = indices_to_remove[:, :-1].clone()
            indices_to_remove[:, 0] = 0

            indices_to_remove = indices_to_remove.scatter(
                1, sorted_indices, indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        next_words = torch.multinomial(probs, num_samples=1).squeeze(1)

        ys = torch.cat([ys, next_words.unsqueeze(1)], dim=1)

        ended = ended | (next_words == eos_idx)

        ys[ended & (ys[:, -1] != eos_idx), -1] = pad_idx

        if ended.all():
            break

    if i == max_len - 2:  # reach max length
        ys[~ended, -1] = eos_idx
        ys[ended, -1] = pad_idx

    return ys


@torch.no_grad
def beam_search(
    model, memory, memory_mask, num_beams, max_len, sos_idx, eos_idx, pad_idx
):
    raise NotImplementedError("TODO")


@torch.no_grad
def translate_sentence(
    sentences: Union[Sequence[str], str],
    model,
    src_tokenizer,
    tgt_tokenizer,
    max_len=config.max_len,
    num_beams=1,
    do_sample=False,
    temperature=None,
    top_k=None,
    top_p=None,
    length_penalty=None,
):
    """
    - *greedy decoding* by calling [`utils.greedy_search`] if `num_beams=1` and
      `do_sample=False`.
    - *multinomial sampling* by calling [`utils.sample`] if `num_beams=1` and
      `do_sample=True`.
    - *beam-search decoding* by calling [`utils.beam_search`] if `num_beams>1` and
      `do_sample=False`.
    """
    if isinstance(sentences, str):
        sentences = [sentences]
    device = next(model.parameters()).device
    sos_idx = tgt_tokenizer.token_to_id(data.SOS_TOKEN)
    eos_idx = tgt_tokenizer.token_to_id(data.EOS_TOKEN)
    pad_idx = tgt_tokenizer.token_to_id(data.PAD_TOKEN)

    src_tensor = torch.LongTensor(
        [encoding.ids for encoding in src_tokenizer.encode_batch(sentences)]
    ).to(device)
    memory = model.encode(src_tensor)
    memory_mask = make_pad_mask(src_tensor, src_tokenizer.token_to_id(data.PAD_TOKEN))

    if num_beams == 1 and do_sample == False:
        tgt_tokens = greedy_search(
            model, memory, memory_mask, max_len, sos_idx, eos_idx, pad_idx
        )
    elif num_beams == 1 and do_sample == True:
        temperature = temperature if temperature is not None else config.temperature
        top_k = top_k if top_k is not None else config.top_k
        top_p = top_p if top_p is not None else config.top_p

        tgt_tokens = sample(
            model,
            memory,
            memory_mask,
            temperature,
            top_k,
            top_p,
            max_len,
            sos_idx,
            eos_idx,
            pad_idx,
        )
    elif num_beams > 1 and do_sample == False:
        length_penalty = (
            length_penalty if length_penalty is not None else config.length_penalty
        )
        tgt_tokens = beam_search(
            model,
            memory,
            memory_mask,
            num_beams,
            length_penalty,
            max_len,
            sos_idx,
            eos_idx,
            pad_idx,
        )
    else:
        raise ValueError("Insupported parameter combination.")

    return ["".join(s) for s in tgt_tokenizer.decode_batch(tgt_tokens.cpu().numpy())]
