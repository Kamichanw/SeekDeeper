from math import ceil
from typing import Optional, Sequence

import datasets
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import config
from modules.bpe import BPETokenizer

pad_idx = 50256


class TokenIDDataset(Dataset):

    def __init__(self, data_source, num_token_per_item: int):
        """
        This class is used to read continuous max_len tokens from data_source.

        Args:
            data_source:
                A Openwebtext dataset where each item contains token ids of a sentence.
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


def _load_openwebtext(tokenizer, loading_ratio, num_proc, splits, **kwargs):
    if not splits is None and splits != ["train"]:
        raise ValueError('Splits must be ["train"] or None.')

    def tokenize(example):
        example["text"] = tokenizer.encode(example["text"])
        return example

    N_DATA_FILES = 21
    # 8013769 rows in total, see https://huggingface.co/datasets/Skylion007/openwebtext
    dataset = (
        datasets.load_dataset(
            str(config.openwebtext_dir),
            split="train",
            num_proc=num_proc,
            num_load_files=ceil(loading_ratio * N_DATA_FILES),
            trust_remote_code=True,
        )
        .select(range(int(8013769 * loading_ratio)))
        .map(tokenize, load_from_cache_file=True, num_proc=num_proc, batched=True)
    )

    return [
        DataLoader(
            TokenIDDataset(dataset, num_token_per_item=config.max_len),
            batch_size=config.batch_size,
            shuffle=True,
        )
    ]


def _load_cst(tokenizer, loading_ratio, num_proc, splits, **kwargs):
    config_name = kwargs.get("config_name", "CN")
    dataset = datasets.load_dataset("cam-cst/cbt", config_name, num_proc=num_proc)

    def collate_fn(batch):
        batch_size = len(batch)
        texts, answer_idx = [], []
        for item in batch:
            # cam-cst/cbt ensures each row in "sentences" column contains 20 sentences
            # and each row in "options" column contains 10 options
            assert len(item["sentences"]) == 20 and len(item["options"]) == 10
            context = " ".join(item["sentences"])
            for i, opt in enumerate(item["options"]):
                if opt == item["answer"]:
                    answer_idx.append(i)
                texts.append(context + item["question"].replace("XXXXX", opt))
        tokens = tokenizer.encode(texts)

        # [batch_size, 10, seq_len], [batch_size]
        return pad_sequence(
            [torch.tensor(tok, dtype=torch.long) for tok in tokens],
            batch_first=True,
            padding_value=pad_idx,
        ).view(batch_size, 10, -1), torch.tensor(answer_idx, dtype=torch.long)

    dataloaders = []

    for split in splits:
        ds = dataset[split]
        subset = ds.select(range(int(loading_ratio * len(ds))))
        dataloaders.append(
            DataLoader(
                subset,
                batch_size=1,
                collate_fn=collate_fn,
                shuffle=split == "train",
            )
        )

    return dataloaders


def load_data(
    name: str,
    loading_ratio: float = 1,
    num_proc: int = 0,
    splits: Sequence[str] = None,
    **kwargs,
):

    tokenizer = BPETokenizer(
        config.openwebtext_dir / "encoder.json",
        config.openwebtext_dir / "vocab.bpe",
    )

    dispatch = {  # _load_* should return a list of dataloader
        "openwebtext": _load_openwebtext,
        "cbt": _load_cst,
    }
    assert (
        name.lower() in dispatch
    ), f"Unsupported dataset, should be one of {list(dispatch.keys())}"
    assert 0 < loading_ratio <= 1

    return tokenizer, *dispatch[name.lower()](
        tokenizer, loading_ratio, num_proc, splits, **kwargs
    )
