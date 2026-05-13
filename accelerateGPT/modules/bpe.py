"""
bpe is short for Byte Pair Encoder. It translates arbitrary utf-8 strings into
sequences of integers, where each integer represents small chunks of commonly
occuring characters. This implementation is based on openai's gpt text_utils.py:
https://github.com/openai/finetune-transformer-lm/blob/master/text_utils.py
"""

import json
import re

import ftfy
import spacy
from tqdm import tqdm


def get_pairs(word):
    """
    Return all bigrams as a set of tuples, of consecutive elements in the iterable word.
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def text_standardize(text):
    """
    fixes some issues the spacy tokenizer had on books corpus
    also does some whitespace standardization
    """
    text = text.replace("—", "-")
    text = text.replace("–", "-")
    text = text.replace("―", "-")
    text = text.replace("…", "...")
    text = text.replace("´", "'")

    # add spaces around all punctuations (-, ~, !, ", ;, ?, +, `,`, ), (, \, /, *, [, ], }, {, |, _)
    # example: "Hi!Kami-chanw" will be converted to "Hi ! Kami - chanw"
    text = re.sub(
        r"""(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)""",
        r" \1 ",
        text,
    )

    # shrink spaces (or add spaces if not space exists) around `\n`
    # exmaple: "Hi\nKamichanw    \n" will be converted to "Hi \n Kamichanw \n "
    text = re.sub(r"\s*\n\s*", " \n ", text)

    # replace all space characters (e.g. `\t`) except `\n` with a single space
    # exmaple: "Hi\tKamichanw   \n" will be converted to "Hi Kamichanw \n "
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()


class BPETokenizer(object):
    """
    mostly a wrapper for a public python bpe tokenizer
    """

    def __init__(self, encoder_path, bpe_path):
        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=["parser", "tagger", "ner", "textcat", "lemmatizer"],
        )
        self.encoder = json.load(open(encoder_path))
        self.decoder = {v: k for k, v in self.encoder.items()}
        merges = open(bpe_path).read().split("\n")[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}
        self.special_tokens = {"</w>"}

    def add_special_tokens(self, new_tokens: list[str]):
        start_idx = len(self.encoder)

        for i, token in enumerate(new_tokens):
            if token in self.encoder:
                raise ValueError(f"Token '{token}' already exists in the encoder.")

            self.encoder[token] = start_idx + i
            self.decoder[start_idx + i] = token

            # no need to update BPE ranks for special tokens as they are not merged
            self.cache[token] = token
        self.special_tokens.update(new_tokens)

    def get_vocab_size(self):
        return len(self.encoder)

    def bpe(self, token):
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:

            # find the next lowest rank bigram that can be merged
            # the lower rank means earlier be merged
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break  # no more bigrams are eligible to be merged
            first, second = bigram

            # we will now replace all occurences of (first, second) in the list of current
            # words into one merged token first_second, in the output list new_words
            new_word = []
            i = 0
            while i < len(word):

                # find the next occurence of first in the sequence of current words
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                # if this occurence is also followed by second, then merge them into one
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            # all occurences of (first, second) have been merged to first_second
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        if word == "\n  </w>":
            word = "\n</w>"
        self.cache[token] = word
        return word

    def token_to_id(self, token: str) -> int:
        return self.encoder.get(token, 0)

    def encode(
        self,
        texts: str | list[str],
        verbose: bool = True,
    ) -> list[list[int]]:
        if not isinstance(texts, list):
            texts = [texts]

        texts_tokens = []
        bar = tqdm(texts, ncols=80, leave=False) if verbose else texts

        for text in bar:
            text = self.nlp(text_standardize(ftfy.fix_text(text)))
            text_tokens = []
            for token in text:
                text_tokens.extend(
                    [
                        self.encoder.get(t, 0)
                        for t in self.bpe(token.text.lower()).split(" ")
                    ]
                )
            texts_tokens.append(text_tokens)
            
        return texts_tokens

    def decode(
        self, bpe_idx: list[list[int]] | list[int], skip_special_tokens=True
    ):
        """lists of integers comes in, a list of string comes out"""
        if not isinstance(bpe_idx[0], list):
            bpe_idx = [bpe_idx]

        texts = []
        for idx in bpe_idx:
            # inverse map the integers to get the tokens
            tokens_merged = [self.decoder[token] for token in idx]
            text = "".join(tokens_merged)
            if skip_special_tokens:
                text = re.sub("|".join(self.special_tokens), " ", text)
            texts.append(text)

        return texts
