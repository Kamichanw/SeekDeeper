from dataclasses import dataclass
from functools import partial
import random
from math import ceil
from typing import List

import datasets
import nltk
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

import config


@dataclass
class PretrainDataSample:
    """
    A data sample for pretraining BERT.
    """

    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    labels: torch.Tensor
    is_next: torch.Tensor


class PretrainDataset(Dataset):
    """
    Prepare the data for pretraining Bert. It converts raw sentences into the format required for pretraining.
    The dataset generate data for the following two tasks:

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    Each sample is in the following format (see Figure 2):
        [CLS] + masked_sentence_A + [SEP] + masked_sentence_B + [SEP]
    """

    def __init__(
        self,
        data_source,
        tokenizer,
    ):
        self.tokenizer = tokenizer
        self.data_source = data_source

    def __len__(self):
        # the last sentence does not have a next sentence
        return len(self.data_source) - 1

    def mask_sentence(self, token_ids: List[int]):
        masked_lm_label = []

        for i, token_id in enumerate(token_ids):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                if prob < 0.8:
                    # 80% randomly change token to mask token
                    token_ids[i] = self.tokenizer.mask_token_id
                elif prob < 0.9:
                    # 10% randomly change token to a random token
                    token_ids[i] = random.randint(0, len(self.tokenizer) - 1)
                else:
                    # 10% stay the same
                    pass

                masked_lm_label.append(token_id)

            else:
                masked_lm_label.append(self.tokenizer.pad_token_id)

        return token_ids, masked_lm_label

    def get_nsp_pair(self, index):
        """
        whenchoosing the sentences A and B for each pretraining example, 50% of the time B is the actual
        next sentence that follows A (labeled as IsNext), and 50% of the time it is a random sentence from
        the corpus (labeled as NotNext).
        """
        sentence_a = self.data_source[index]
        is_next = random.random() > 0.5
        if is_next:
            sentence_b = self.data_source[index + 1]
        else:
            sentence_b = self.data_source[random.randint(0, len(self.data_source) - 1)]

        # we suppose that the input is a dictionary with a key "text"
        # otherwise, check _load_* or dataset preparation
        assert isinstance(sentence_a, dict) and isinstance(sentence_b, dict)
        assert "text" in sentence_a and "text" in sentence_b
        return sentence_a["text"], sentence_b["text"], is_next

    def __getitem__(self, index):
        sentence_a, sentence_b, is_next = self.get_nsp_pair(index)
        masked_sentence_a, sentence_a_labels = self.mask_sentence(sentence_a)
        masked_sentence_b, sentence_b_labels = self.mask_sentence(sentence_b)

        input_ids = (
            [self.tokenizer.cls_token_id]
            + masked_sentence_a
            + [self.tokenizer.sep_token_id]
            + masked_sentence_b
            + [self.tokenizer.sep_token_id]
        )

        labels = (
            [self.tokenizer.pad_token_id]
            + sentence_a_labels
            + [self.tokenizer.pad_token_id]
            + sentence_b_labels
            + [self.tokenizer.pad_token_id]
        )

        segment_label = [0] * (1 + len(masked_sentence_a) + 1) + [1] * (
            len(masked_sentence_b) + 1
        )

        return PretrainDataSample(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            token_type_ids=torch.tensor(segment_label, dtype=torch.long),
            labels=torch.tensor(labels, dtype=torch.long),
            is_next=torch.tensor(is_next, dtype=torch.long),
        )


def collate_fn(batch: List[PretrainDataSample], pad_token_id: int):
    input_ids = pad_sequence(
        [item.input_ids for item in batch], padding_value=pad_token_id, batch_first=True
    )
    token_type_ids = pad_sequence(
        [item.token_type_ids for item in batch],
        padding_value=0,
        batch_first=True,
    )
    labels = pad_sequence(
        [item.labels for item in batch], padding_value=pad_token_id, batch_first=True
    )
    is_next = pad_sequence(
        [item.is_next for item in batch],
        padding_value=False,
        batch_first=True,
    )

    return PretrainDataSample(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        labels=labels,
        is_next=is_next,
    )


def _load_wikipedia(tokenizer, loading_ratio, num_proc, splits):
    if not splits is None and splits != ["train"]:
        raise ValueError('Splits must be ["train"] or None.')

    def tokenize(example):
        if not isinstance(example["text"], list):
            text_list = [example["text"]]
        else:
            text_list = example["text"]
        sentences = [nltk.sent_tokenize(t) for t in text_list]
        token_ids = tokenizer(
            [t for s in sentences for t in s],
            padding="longest",
            truncation=True,
            max_length=config.max_len,
        ).input_ids
        return {"text": token_ids}

    dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    subset = dataset.select(range(int(loading_ratio * len(dataset)))).map(
        tokenize,
        load_from_cache_file=True,
        num_proc=num_proc,
        batched=True,
        remove_columns=dataset.column_names,
    )

    return [
        DataLoader(
            PretrainDataset(subset, tokenizer=tokenizer),
            batch_size=config.PretrainingConfig.batch_size,
            collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
            shuffle=True,
        )
    ]


def _load_bookcorpus(tokenizer, loading_ratio, num_proc, splits):
    if not splits is None and splits != ["train"]:
        raise ValueError('Splits must be ["train"] or None.')

    def tokenize(example):
        example["text"] = tokenizer(
            example["text"],
            padding="longest",
            truncation=True,
            max_length=config.max_len,
        ).input_ids
        return example

    # 10 files in total, but we may just use part of them
    URLS = [
        f"https://hf-mirror.com/datasets/bookcorpus/bookcorpus/resolve/refs%2Fconvert%2Fparquet/plain_text/train/000{i}.parquet?download=true"
        for i in range(ceil(loading_ratio * 10))
    ]

    dl_manager = datasets.DownloadManager("bookcorpus")
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
            PretrainDataset(dataset, tokenizer=tokenizer),
            batch_size=config.PretrainingConfig.batch_size,
            collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
            shuffle=True,
        )
    ]


def _load_sst2(tokenizer, loading_ratio, num_proc, splits):
    all_splits = ["train", "validation", "test"]
    if splits is None:
        splits = all_splits
    elif not set(splits).issubset(all_splits):
        raise ValueError(f"Splits should only contain some of {all_splits}")

    dataset = datasets.load_dataset("glue", "sst2", num_proc=num_proc)

    def collate_fn(batch):
        sentences, labels = [], []
        for item in batch:
            sentences.append(tokenizer.cls_token + item["sentence"])
            labels.append(item["label"])
        tokens = tokenizer(
            sentences,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=config.max_len,
        ).input_ids
        return pad_sequence(
            tokens, batch_first=True, padding_value=tokenizer.pad_token_id
        ), torch.tensor(labels, dtype=torch.long)

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
    splits: list = None,
    **kwargs,
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    dispatch = {  # _load_* should return a list of dataloader
        "sst2": _load_sst2,
        "bookcorpus": _load_bookcorpus,
        "wikipedia": _load_wikipedia,
    }

    if name.lower() not in dispatch:
        raise ValueError(
            f"Unsupported dataset '{name}'. Supported datasets are: {list(dispatch.keys())}"
        )

    if not (0 < loading_ratio <= 1):
        raise ValueError("Loading ratio should be between 0 and 1")

    return tokenizer, *dispatch[name.lower()](
        tokenizer, loading_ratio, num_proc, splits, **kwargs
    )
