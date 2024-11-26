import os
from pathlib import Path

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    str(i) for i in range(torch.cuda.device_count())
)

torch.manual_seed(3407)

# training setting
batch_size = 32
accumulate_grad_batches = max(1, 512 // batch_size)

# Although the paper's B.1.1 requires the following steps for fine-tuning,
# there are still too many training steps.
# Therefore, we use an alternate setting that is consistent with Table 6.
total_steps = {
    "imagenet": 20000,
    "cifar100": 10000,
    "cifar10": 10000,
    "oxford-iiit-pets": 500,
    "oxford-flowers-102": 500,
    "vtab": 2500,
}
# 7 epcohs for ViT-B/32, ViT-B/16, ViT-L/32, ViT-L/16
num_epochs = 7
resolution = 384

weight_decay = 0
momentum = 0.9

base_lr = {
    "imagenet": [0.003, 0.01, 0.03, 0.06],
    "cifar100": [0.001, 0.003, 0.01, 0.03],
    "cifar10": [0.001, 0.003, 0.01, 0.03],
    "oxford-iiit-pets": [0.001, 0.003, 0.01, 0.03],
    "oxford-flowers-102": [0.001, 0.003, 0.01, 0.03],
    "vtab": [0.1],
}


# path
base_dir = Path(__file__).parent.resolve()
checkpoint_dir = base_dir / "checkpoints"
dataset_dir = base_dir / "datasets"
