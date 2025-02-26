import os
from dataclasses import dataclass
from pathlib import Path

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    str(i) for i in range(torch.cuda.device_count())
)

torch.manual_seed(3407)

# model parameter setting
max_len = 512
hidden_size = 768
num_hidden_layers = 12
num_attention_heads = 12
vocab_size = 40478
dropout = 0.1


@dataclass
class PretrainConfig:
    n_epoch = 100
    batch_size = 16
    accumulate_grad_batches = 4
    lr = 2.5e-4
    warmup_steps = 2000


@dataclass
class FinetuningConfig:
    n_epoch = 3
    batch_size = 32
    accumulate_grad_batches = 1
    lr = 6.25e-5
    clf_loss_weight = 0.5
    warmup_steps = 0.2 / 100


# path
base_dir = Path(__file__).parent.resolve()
checkpoint_dir = base_dir / "checkpoints"
bookcorpus_dir = base_dir / "datasets" / "bookcorpus"

# if you downloaded pretrained gpt in an automatical manner, set pretrained_dir to "openai-community/openai-gpt"
# otherwise, you can obtain pretrained model with huggingface-cli, see README.md appendix
pretrained_dir = checkpoint_dir / "gpt"

os.makedirs(checkpoint_dir, exist_ok=True)

# optimizer args
weight_decay = 1e-2
clip = 1

# inference
num_beams = 3
top_k = 30
top_p = 0.7
temperature = 1.0
length_penalty = 0.7
