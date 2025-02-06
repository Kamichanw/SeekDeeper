import os
from dataclasses import dataclass
from pathlib import Path

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    str(i) for i in range(torch.cuda.device_count())
)

torch.manual_seed(3407)

# basic setting
vocab_size = 30522
encoding = "utf-8"
min_freq = 1

# training setting
batch_size = 32
epochs = 11
num_workers = 5

# dataset and vocabulary paths
base_dir = Path(__file__).parent.resolve()
checkpoint_dir = base_dir / "checkpoints"

train_dataset = base_dir / "dataset"/"corpus.txt"
test_dataset = None  # set to the path of the test dataset if available

trained_path = checkpoint_dir / "bert_self_trained"
pretrained_path = "bert-base-uncased"

vocab_path = base_dir/"output/vocab"

# model hyperparameters
hidden_size = 768
num_layers = 12
attention_heads = 12
sequence_length = 512

# cuda and logging configurations
with_cuda = True
cuda_devices = None  # list of cuda device ids
log_freq = 5

# memory options
corpus_lines = None  # total number of lines in corpus
on_memory = True

# optimizer parameters
learning_rate = 1e-4
adam_weight_decay = 0.01
adam_beta1 = 0.9
adam_beta2 = 0.999
warmup_steps = 10000
total_steps = 1000000


@dataclass
class FinetuningConfig:
    n_epoch = 3
    batch_size = 32
    lr = 5e-5
    weight_decay = 0.01
    warmup_steps = 10000




