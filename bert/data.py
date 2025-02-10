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
    ''' This is for processing the corpus  '''
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True, loading_ratio = 0.00001):
        self.vocab = vocab
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
                for i, line in tqdm.tqdm(enumerate(f), desc="Loading Dataset", total=num_load_lines):
                    if i >= num_load_lines:  # Stop once the required number of lines is loaded
                        break
                    if line.strip(): # not empty
                        self.lines.append(line[:-1].split("\t"))
                print(f"length of self.lines: {len(self.lines)}")
                self.corpus_lines = len(self.lines)

        # if not on_memory:
        #     self.file = open(corpus_path, "r", encoding=encoding)
        #     self.random_file = open(corpus_path, "r", encoding=encoding)
        #
        #     for _ in range(random.randint(0, self.corpus_lines if self.corpus_lines < 1000 else 1000)):
        #         self.random_file.__next__()
        if not on_memory:
            # If not loading into memory, we still respect the loading_ratio and limit the lines read.
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

            # Read only the specified percentage of lines based on loading_ratio
            # Instead of random skipping, we now limit the reading to num_load_lines
            self.lines = []
            for i, line in tqdm.tqdm(enumerate(self.file), desc="Loading Dataset", total=num_load_lines):
                if i >= num_load_lines:  # Stop once the required number of lines is read
                    break
                if line.strip():  # Skip empty lines
                    self.lines.append(line[:-1].split("\t"))
            self.corpus_lines = len(self.lines)
            print(f"Loaded {self.corpus_lines} lines into memory with loading_ratio: {loading_ratio}")


    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t1, t2, is_next_label = self.random_sent(item)
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        t2 = t2_random + [self.vocab.eos_index]

        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        t2_label = t2_label + [self.vocab.pad_index]

        # segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        # 0 for sentence A, 1 for sentence B
        segment_label = ([0 for _ in range(len(t1))] + [1 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    # random masking
    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)

        return tokens, output_label

    # random sentence pair selection (for NSP)
    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    # def get_corpus_line(self, item):
    #     print("get_corpus_line:item:", item)
    #     if self.on_memory:
    #         return self.lines[item][0], self.lines[item][1]
    #     else:
    #         line = self.file.__next__()
    #         if line is None:
    #             self.file.close()
    #             self.file = open(self.corpus_path, "r", encoding=self.encoding)
    #             line = self.file.__next__()
    #
    #         t1, t2 = line[:-1].split("\t")
    #         print("get_corpus_line:t1, t2:", t1, t2)
    #         return t1, t2

    # get one line from corpus, which contains two sentences
    def get_corpus_line(self, item):
        # print(f"get_corpus_line:item:{item}\n")
        if self.on_memory:
            t1, t2 = self.lines[item]
            if not t1 or not t2:  # Skip empty lines
                return self.get_corpus_line(item + 1)  # Try next line, but may cause some duplications(quite little)
            return t1, t2
        else:
            line = self.file.__next__()
            while not line.strip():  # skip empty lines
                line = self.file.__next__()

            parts = line[:-1].split("\t")
            if len(parts) < 2:  # if a line has less than 2 parts, skip it
                return self.get_corpus_line(item + 1)  # get next line

            t1, t2 = parts
            return t1, t2

    # def get_random_line(self):
    #     if self.on_memory:
    #         return self.lines[random.randrange(len(self.lines))][1]
    #
    #     line = self.file.__next__()
    #     if line is None:
    #         self.file.close()
    #         self.file = open(self.corpus_path, "r", encoding=self.encoding)
    #         for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
    #             self.random_file.__next__()
    #         line = self.random_file.__next__()
    #     return line[:-1].split("\t")[1]

    # choose one random line from corpus
    def get_random_line(self):
        if self.on_memory:
            while True:
                # Randomly select a line
                line = self.lines[random.randrange(len(self.lines))]
                if len(line) > 1 and line[1].strip():  # Ensure the second part (t2) is non-empty
                    return line[1]  # Return the second part of the line (t2)
        else:
            while True:
                line = self.file.__next__()
                if line is None:  # If line is None, reset file reading
                    self.file.close()
                    self.file = open(self.corpus_path, "r", encoding=self.encoding)
                    for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                        self.random_file.__next__()  # Skip random lines
                    line = self.random_file.__next__()

                parts = line[:-1].split("\t")  # Split the line into two parts
                if len(parts) > 1 and parts[1].strip():  # Ensure the second part (t2) is non-empty
                    return parts[1]  # Return the second part of the line (t2)


class TorchVocab(object):
    """Defines a vocabulary object that will be used to numericalize a field.
    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
    """

    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>', '<oov>'],
                 vectors=None, unk_init=None, vectors_cache=None):
        """Create a Vocab object from a collections.Counter.
        Arguments:
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary in addition to an <unk>
                token. Default: ['<pad>']
            vectors: One of either the available pretrained vectors
                or custom pretrained vectors (see Vocab.load_vectors);
                or a list of aforementioned vectors
            unk_init (callback): by default, initialize out-of-vocabulary word vectors
                to zero vectors; can be any function that takes in a Tensor and
                returns a Tensor of the same size. Default: torch.Tensor.zero_
            vectors_cache: directory for cached vectors. Default: '.vector_cache'
        """
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list(specials)
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]

        max_size = None if max_size is None else max_size + len(self.itos)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1


class Vocab(TorchVocab):
    def __init__(self, counter, max_size=None, min_freq=1):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        super().__init__(counter, specials=["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"],
                         max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentece, seq_len, with_eos=False, with_sos=False) -> list:
        pass

    def from_seq(self, seq, join=False, with_pad=False):
        pass

    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


# Building Vocab with text files
class WordVocab(Vocab):
    def __init__(self, texts, max_size=None, min_freq=1):
        print("Building Vocab")
        counter = Counter()
        for line in tqdm.tqdm(texts):
            if isinstance(line, list):
                words = line
            else:
                words = line.replace("\n", "").replace("\t", "").split()

            for word in words:
                counter[word] += 1
        super().__init__(counter, max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        if isinstance(sentence, str):
            sentence = sentence.split()

        seq = [self.stoi.get(word, self.unk_index) for word in sentence]

        if with_eos:
            seq += [self.eos_index]  # this would be index 1
        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        words = [self.itos[idx]
                 if idx < len(self.itos)
                 else "<%d>" % idx
                 for idx in seq
                 if not with_pad or idx != self.pad_index]

        return " ".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab':
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(vocab_path, "rb") as f:
            return pickle.load(f)


def build():
    if not config.vocab_path.exists():
        print("Building Vocab")
        with open(config.train_dataset, "r", encoding=config.encoding) as f:
            vocab = WordVocab(f, max_size=config.vocab_size, min_freq=config.min_freq)
        print("VOCAB SIZE:", len(vocab))

        # test: print first 100 words of vocab
        print("VOCAB CONTENT:", vocab.itos[:100])

        # Save Vocab
        vocab.save_vocab(config.vocab_path)
    else:
        print("Vocab already exists! If you want to rebuild it, please delete the existing vocab file.")


# load bert-pertrain dataset: bookcorpus and English wikipedia
def split_into_sentences(text): # with nltk
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


def load_bookcorpus_wikipedia(book_loading_ratio=0.1, wiki_loading_ratio=1/41):

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
            features = Features({
                'id': Value('string'),
                'url': Value('string'),
                'title': Value('string'),
                'text': Value('string')
            })
            wiki_urls = [
                f"https://huggingface.co/datasets/wikimedia/wikipedia/resolve/refs%2Fconvert%2Fparquet/20231101.en/train/000{i}.parquet" # use version-20231101.en for Wikipedia(latest in wikimedia in huggingface)
                for i in range(num_wiki_files)
            ]
            print("Loading Wikipedia URLs:", wiki_urls)
            # extract train data
            wikipedia_ds = load_dataset("parquet", data_files=wiki_urls, split="train", features=features)
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
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

    # tokenization
    tokenized_train = train_data.map(tokenize_function, batched=True)
    tokenized_valid = valid_data.map(tokenize_function, batched=True)

    # DataLoader format
    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['label'] for item in batch]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long)
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


def load_data(name: str, loading_ratio: float = 1, num_proc: int = 0, splits: list = None, **kwargs):
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
        raise ValueError(f"Unsupported dataset '{name}'. Supported datasets are: {list(dispatch.keys())}")

    if not (0 < loading_ratio <= 1):
        raise ValueError("Loading ratio should be between 0 and 1")

    # 调用对应的加载函数
    return tokenizer, *dispatch[name.lower()](tokenizer, loading_ratio, num_proc, splits, **kwargs)
