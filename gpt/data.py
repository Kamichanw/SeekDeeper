from math import ceil
from typing import Sequence

import datasets
from datasets import DownloadManager
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import config
from modules.bpe import BPETokenizer

SOS_TOKEN = "<start>"
CLF_TOKEN = "<extract>"

special_tokens = [SOS_TOKEN, CLF_TOKEN]
pad_idx = 0  # reuse unk as pad token


class TokenIDDataset(Dataset):

    def __init__(self, data_source, num_token_per_item: int):
        """
        This class is used to read continuous max_len tokens from data_source.

        Args:
            data_source:
                A BookCorpus dataset where each item contains token ids of a sentence.
            num_token_per_item (int):
                The number of token ids of an item.
        """
        super().__init__()
        self.data_source = data_source
        self.num_token_per_item = num_token_per_item

    def __getitem__(self, index):
        rest_length = self.num_token_per_item
        sequence = []
        while rest_length > 0:
            token_ids = self.data_source[index]["text"]
            sequence.extend(token_ids[:rest_length])
            rest_length -= len(token_ids)
            index += 1
        return torch.tensor(sequence, dtype=torch.long)

    def __len__(self):
        accumulate_length = 0
        for i, example in enumerate(reversed(self.data_source)):
            accumulate_length += len(example["text"])
            if accumulate_length >= self.num_token_per_item:
                return len(self.data_source) - i
        return 0


def _load_bookcorpus(tokenizer, loading_ratio, num_proc, splits):
    if not splits is None and splits != ["train"]:
        raise ValueError('Splits must be ["train"] or None.')

    def tokenize(example):
        example["text"] = tokenizer.encode(example["text"], verbose=False)
        return example

    # 10 files in total, but we may just use part of them
    URLS = [
        f"https://hf-mirror.com/datasets/bookcorpus/bookcorpus/resolve/refs%2Fconvert%2Fparquet/plain_text/train/000{i}.parquet?download=true"
        for i in range(ceil(loading_ratio * 10))
    ]

    dl_manager = DownloadManager("bookcorpus")
    paths = dl_manager.download(URLS)
    print("Downloaded at ", paths)

    # 74004228 rows in total, see https://huggingface.co/datasets/bookcorpus/bookcorpus
    dataset = (
        datasets.load_dataset(
            "parquet", data_files=paths, split="train", num_proc=num_proc
        )
        .select(range(int(loading_ratio * 74004228)))
        .map(tokenize, load_from_cache_file=True, num_proc=num_proc, batched=True)
    )

    return [
        DataLoader(
            TokenIDDataset(dataset, num_token_per_item=config.max_len),
            batch_size=config.PretrainConfig.batch_size,
            shuffle=True,
        )
    ]


def _load_sst2(tokenizer: BPETokenizer, loading_ratio, num_proc, splits):
    all_splits = ["train", "validation", "test"]
    if splits is None:
        splits = all_splits
    elif not set(splits).issubset(all_splits):
        raise ValueError(f"Splits should only contain some of {all_splits}")

    tokenizer.add_special_tokens(special_tokens)

    def collate_fn(batch):
        sentences, labels = [], []
        for item in batch:
            sentences.append(SOS_TOKEN + item["sentence"] + CLF_TOKEN)
            labels.append(item["label"])
        tokens = tokenizer.encode(
            sentences,
            verbose=False,
        )
        tensors = [torch.tensor(tok, dtype=torch.long) for tok in tokens]
        return pad_sequence(
            tensors, batch_first=True, padding_value=pad_idx
        ), torch.tensor(labels, dtype=torch.long)

    dataset = datasets.load_dataset("stanfordnlp/sst2", num_proc=num_proc)

    dataloaders = []

    for split in splits:
        ds = dataset[split]
        subset = ds.select(range(int(loading_ratio * len(ds))))
        dataloaders.append(
            DataLoader(
                subset,
                config.FinetuningConfig.batch_size,
                collate_fn=collate_fn,
                shuffle=split == "train",
            )
        )

    return dataloaders


def load_data(
    name: str,
    loading_ratio: float = 1,
    num_proc: int = 1,
    splits: Sequence[str] = None,
):
    dispatch = {  # _load_* should return a list of dataloader
        "bookcorpus": _load_bookcorpus,
        "sst2": _load_sst2,
    }
    assert (
        name.lower() in dispatch
    ), f"Unsupported dataset, should be one of {list(dispatch.keys())}"
    assert 0 < loading_ratio <= 1

    tokenizer = BPETokenizer(
        config.bookcorpus_dir / "encoder_bpe_40000.json",
        config.bookcorpus_dir / "vocab_40000.bpe",
    )

    return tokenizer, *dispatch[name.lower()](
        tokenizer, loading_ratio, num_proc, splits
    )
