import os
from dataclasses import dataclass
from pathlib import Path

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    str(i) for i in range(torch.cuda.device_count())
)
os.environ["END_POINT"] = "hf-mirror.com"

torch.manual_seed(3407)

# dataset and vocabulary paths
base_dir = Path(__file__).parent.resolve()
checkpoint_dir = base_dir / "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

pretrained_path = "bert-base-uncased"


# model hyperparameters
hidden_size = 768
num_layers = 12
attention_heads = 12
max_len = 512
dropout = 0.1
intermediate_size = 3072


@dataclass
class PretrainConfig:
    n_epoch = 2
    batch_size = 32
    lr = 1e-4
    weight_decay = 0.01
    warmup_steps = 10000


@dataclass
class FinetuningConfig:
    n_epoch = 3
    batch_size = 32
    lr = 4e-5
    weight_decay = 0.01
    warmup_steps = 10000
