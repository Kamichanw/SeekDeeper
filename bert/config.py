import os
from pathlib import Path

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    str(i) for i in range(torch.cuda.device_count())
)

torch.manual_seed(3407)

# basic setting
vocab_size = 5000
encoding = "utf-8"
min_freq = 1

# training setting
batch_size = 64
epochs = 5
num_workers = 5


# dataset and vocabulary paths
base_dir = Path(__file__).parent.resolve()
checkpoint_dir = base_dir / "checkpoints"

train_dataset = base_dir / "datasets"/"corpus.txt"
test_dataset = None  # set to the path of the test dataset if available
vocab_path = base_dir/"output/vocab"
output_path = base_dir/"output/bert.model"


# model hyperparameters
hidden_size = 256
num_layers = 8
attention_heads = 8
sequence_length = 20

# cuda and logging configurations
with_cuda = True
cuda_devices = None  # list of cuda device ids
log_freq = 10

# memory options
corpus_lines = None  # total number of lines in corpus
on_memory = True

# optimizer parameters
learning_rate = 1e-3
adam_weight_decay = 0.01
adam_beta1 = 0.9
adam_beta2 = 0.999

# inference


