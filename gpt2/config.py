import os
from pathlib import Path

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    str(i) for i in range(torch.cuda.device_count())
)

torch.manual_seed(3407)

batch_size = 8
accumulate_grad_batches = 64
# gpt-2
max_len = 1024
vocab_size = 50257
d_model = 768
n_layer = 12
n_head = 12
dropout = 0.1


# note that the original GPT-2 paper doesn't specify details about training hyper parameters
# some of the following parameters are from https://github.com/karpathy/nanoGPT/blob/master/train.py
lr = 6e-4  # max learning rate
n_epoch = 10
weight_decay = 1e-1
betas = (0.9, 0.95)
clip = 1.0  # clip gradients at this value, or disable if == 0.0
min_lr = 6e-5

# path
base_dir = Path(__file__).parent.resolve()
checkpoint_dir = base_dir / "checkpoints"
pretrained_dir = checkpoint_dir / "gpt2"
dataset_dir = base_dir / "datasets"
openwebtext_dir = dataset_dir / "openwebtext"
