import os
from pathlib import Path

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    str(i) for i in range(torch.cuda.device_count())
)

torch.manual_seed(3407)

# model parameter setting (Transformer base)
max_len = 512
hidden_size = 512
num_hidden_layers = 6
num_attention_heads = 8
ffn_hidden = 2048
dropout = 0.1

# training setting
batch_size = 32
accumulate_grad_batches = 16
epochs = 20
eps_ls = 0.1  # eps for label smoothing
warmup_step = 4000
clip = 1

# optimizer parameter setting
betas = (0.9, 0.98)
adam_eps = 1e-9

# path
base_dir = Path(__file__).parent.resolve()
checkpoint_dir = base_dir / "checkpoints"
dataset_dir = base_dir / "datasets" / "IWSLT17"

os.makedirs(checkpoint_dir, exist_ok=True)

# inference
num_beams = 3
top_k = 30
top_p = 0.7
temperature = 1.0
length_penalty = 0.7
