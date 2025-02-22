import os
import pickle
import random
from collections import Counter
from math import ceil

import nltk
import torch
import tqdm
from datasets import load_dataset, Features, Value
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import config

class BERTDataset(Dataset):
    """This is for processing the corpus"""
    def __init__(
        self,
        corpus_path,
        tokenizer,
        seq_len,
        encoding="utf-8",
        corpus_lines=None,
        on_memory=True,
        loading_ratio=0.00001,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.loading_ratio = loading_ratio  # loading ratio for corpus lines

        with open(corpus_path, "r", encoding=encoding) as f:

            num_lines = sum(1 for _ in f)
            num_load_lines = int(num_lines * self.loading_ratio)
            print(f"num_load_lines: {num_load_lines}")

            #  Reset file pointer to the beginning
            f.seek(0)

            if self.corpus_lines is None and not on_memory:
                self.corpus_lines = num_load_lines

            if on_memory:  # to save memory, just load part of the corpus to train
                self.lines = []
                for i, line in tqdm.tqdm(
                    enumerate(f), desc="Loading Dataset", total=num_load_lines
                ):
                    if (
                        i >= num_load_lines
                    ):  # Stop once the required number of lines is loaded
                        break
                    if line.strip():  # not empty
                        self.lines.append(line[:-1].split("\t"))
                print(f"length of self.lines: {len(self.lines)}")
                self.corpus_lines = len(self.lines)

        if not on_memory:
            # If not loading into memory, we still respect the loading_ratio and limit the lines read.
            self.file = open(corpus_path, "r", encoding=encoding)

            self.lines = []
            for i, line in tqdm.tqdm(
                enumerate(self.file), desc="Loading Dataset", total=num_load_lines
            ):
                if (
                    i >= num_load_lines
                ):  # Stop once the required number of lines is read
                    break
                if line.strip():  # Skip empty lines
                    self.lines.append(line[:-1].split("\t"))
            self.corpus_lines = len(self.lines)
            print(
                f"Loaded {self.corpus_lines} lines into memory with loading_ratio: {loading_ratio}"
            )

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t1, t2, is_next_label = self.random_sent(item)
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # Add special tokens
        t1_tokens = [self.tokenizer.cls_token_id] + t1_random + [self.tokenizer.sep_token_id]
        t2_tokens = t2_random + [self.tokenizer.sep_token_id]

        # Add padding if necessary
        bert_input = (t1_tokens + t2_tokens)[:self.seq_len]
        padding = [self.tokenizer.pad_token_id for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding)

        # Attention mask (1 for real tokens, 0 for padding tokens)
        attention_mask = [1 if token != self.tokenizer.pad_token_id else 0 for token in bert_input]

        # Labels for masked tokens
        t1_label = [self.tokenizer.pad_token_id] + t1_label + [self.tokenizer.pad_token_id]
        t2_label = t2_label + [self.tokenizer.pad_token_id]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        # Segment labels
        segment_label = ([0 for _ in range(len(t1_tokens))] + [1 for _ in range(len(t2_tokens))])[:self.seq_len]

        # Pad the labels
        bert_label.extend([self.tokenizer.pad_token_id] * (self.seq_len - len(bert_label)))
        segment_label.extend([0] * (self.seq_len - len(segment_label)))

        output = {
            "bert_input": bert_input,
            "bert_attention_mask": attention_mask,
            "bert_label": bert_label,
            "segment_label": segment_label,
            "is_next": is_next_label,
        }

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        """ mask tokens, return tokens_ids and labels"""
        tokens = sentence.split()
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.choice(list(self.tokenizer.vocab.values()))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.tokenizer.convert_tokens_to_ids(token)

                output_label.append(self.tokenizer.convert_tokens_to_ids(token))

            else:
                tokens[i] = self.tokenizer.convert_tokens_to_ids(token)
                output_label.append(0)

        return tokens, output_label

    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        if self.on_memory:
            t1, t2 = self.lines[item]
            if not t1 or not t2:
                return self.get_corpus_line(item + 1)
            return t1, t2
        else:
            line = self.file.__next__()
            while not line.strip():
                line = self.file.__next__()

            parts = line[:-1].split("\t")
            if len(parts) < 2:
                return self.get_corpus_line(item + 1)

            t1, t2 = parts
            return t1, t2

    def get_random_line(self):
        if self.on_memory:
            while True:
                line = self.lines[random.randrange(len(self.lines))]
                if len(line) > 1 and line[1].strip():
                    return line[1]
        else:
            while True:
                line = self.file.__next__()
                if line is None:
                    self.file.close()
                    self.file = open(self.corpus_path, "r", encoding=self.encoding)
                    for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                        self.random_file.__next__()
                    line = self.random_file.__next__()

                parts = line[:-1].split("\t")
                if len(parts) > 1 and parts[1].strip():
                    return parts[1]



# load bert-pertain dataset: book corpus and English wikipedia
def split_into_sentences(text):  # with nltk
    return nltk.sent_tokenize(text)


def generate_sentence_pairs(sentences):
    pairs = []
    for i in range(0, len(sentences) - 1, 2):
        # pair ont sentence with its next sentence
        if i + 1 < len(sentences):
            # ensure sentences are not empty to avoid empty lines
            sentence1 = sentences[i].strip()
            sentence2 = sentences[i + 1].strip()
            if sentence1 and sentence2:  # Skip empty sentences
                pairs.append(f"{sentence1}\t{sentence2}")
    return pairs


def save_to_file(pairs, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            # Ensure the pair is not empty or just whitespace
            if pair.strip():
                f.write(pair + "\n")


def load_bookcorpus_wikipedia(book_loading_ratio=0.1, wiki_loading_ratio=1 / 41):

    import nltk
    nltk.download("punkt")
    nltk.download('punkt_tab')

    bookcorpus_path = config.base_dir / "dataset" / "bookcorpus_sentences.txt"
    wikipedia_path = config.base_dir / "dataset" / "wikipedia_sentences.txt"

    if not config.train_dataset.exists():
        if not bookcorpus_path.exists():
            # load BookCorpus, only load 10% data
            num_book_files = ceil(10 * book_loading_ratio)
            book_urls = [
                f"https://hf-mirror.com/datasets/bookcorpus/bookcorpus/resolve/refs%2Fconvert%2Fparquet/plain_text/train/000{i}.parquet?download=true"
                for i in range(num_book_files)
            ]
            print("Loading BookCorpus URLs:", book_urls)

            bookcorpus_ds = load_dataset("parquet", data_files=book_urls, split="train")
            bookcorpus_texts = [example["text"] for example in bookcorpus_ds]

            # split into sentences
            bookcorpus_sentences = []
            for text in bookcorpus_texts:
                bookcorpus_sentences.extend(split_into_sentences(text))

            bookcorpus_pairs = generate_sentence_pairs(bookcorpus_sentences)

            save_to_file(bookcorpus_pairs, bookcorpus_path)

        if not wikipedia_path.exists():
            # load English Wikipedia,only load 1/41 data
            num_wiki_files = ceil(41 * wiki_loading_ratio)

            # data structure of wikipedia
            features = Features(
                {
                    "id": Value("string"),
                    "url": Value("string"),
                    "title": Value("string"),
                    "text": Value("string"),
                }
            )
            wiki_urls = [
                f"https://huggingface.co/datasets/wikimedia/wikipedia/resolve/refs%2Fconvert%2Fparquet/20231101.en/train/000{i}.parquet"  # use version-20231101.en for Wikipedia(latest in wikimedia in huggingface)
                for i in range(num_wiki_files)
            ]
            print("Loading Wikipedia URLs:", wiki_urls)
            # extract train data
            wikipedia_ds = load_dataset(
                "parquet", data_files=wiki_urls, split="train", features=features
            )
            wikipedia_texts = [example["text"] for example in wikipedia_ds]

            wikipedia_sentences = []
            for text in wikipedia_texts:
                wikipedia_sentences.extend(split_into_sentences(text))

            wikipedia_pairs = generate_sentence_pairs(wikipedia_sentences)

            save_to_file(wikipedia_pairs, wikipedia_path)

        with open(config.train_dataset, "w", encoding="utf-8") as outfile:
            with open(bookcorpus_path, "r", encoding="utf-8") as infile1:
                outfile.write(infile1.read())
            with open(wikipedia_path, "r", encoding="utf-8") as infile2:
                outfile.write(infile2.read())


def _load_sst2(tokenizer, loading_ratio, num_proc, splits, **kwargs):
    """
    load and return SST-2 's DataLoader
    :param tokenizer:  text process, BERT tokenizer
    :param loading_ratio: ratio of loading data
    :param num_proc: number of processes for data loading
    :param splits: split of data (such as "train", "validation"）
    :param kwargs: other parameters
    :return: DataLoaders' list
    """

    # check splits
    if not splits or splits != ["train", "validation"]:
        raise ValueError('Splits must be ["train", "validation"] or None.')

    # load with function from datasets library
    dataset = load_dataset("glue", "sst2", num_proc=num_proc)
    # compute subset size
    total_samples = len(dataset["train"])  # 80137 rows
    subset_size = int(loading_ratio * total_samples)

    train_size = len(dataset["train"])
    valid_size = len(dataset["validation"])

    # make sure subset size is not larger than the original size
    train_subset_size = min(subset_size, train_size)
    valid_subset_size = min(subset_size, valid_size)

    train_data = dataset["train"].select(range(train_subset_size))
    valid_data = dataset["validation"].select(range(valid_subset_size))

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"], padding="max_length", truncation=True, max_length=128
        )

    # tokenization
    tokenized_train = train_data.map(tokenize_function, batched=True)
    tokenized_valid = valid_data.map(tokenize_function, batched=True)

    # DataLoader format
    def collate_fn(batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["label"] for item in batch]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    dataloaders = []

    # create DataLoader for each split
    for split_data in [(tokenized_train, "train"), (tokenized_valid, "validation")]:
        tokenized_split, split_name = split_data
        dataloaders.append(
            DataLoader(
                tokenized_split,
                batch_size=config.FinetuningConfig.batch_size,
                collate_fn=collate_fn,
                shuffle=split_name == "train",
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
    """
    Load different datasets
    :param name: name of dataset
    :param loading_ratio: ratio of loading data
    :param num_proc: number of processes for data loading
    :param splits: split of data (such as "train", "validation"）
    :param kwargs: other parameters
    :return: tokenizer and dataloaders
    """
    # Load model directly
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    dispatch = {
        "sst2": _load_sst2,  # SST-2 load
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
