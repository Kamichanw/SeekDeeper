[📖中文 ReadMe](./README_zh.md)
## Introduction

In this BERT implementation, we will demonstrate how to conduct pre-training on the  [BookCorpus](https://huggingface.co/datasets/bookcorpus/bookcorpus) and [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) datasets, load the official pre-trained weights provided by Hugging Face, and fine-tune on the [Stanford Sentiment Treebank (SST-2)](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf) dataset to reproduce the results reported in the original paper.

## Model details

### Key differences with GPT

In fact, many of the design decisions in BERT were intentionally made to make it as close to GPT as possible so that the two methods could be minimally compared.  For instance, BERT_base matches GPT in model size (e.g., layers, attention heads, and hidden dimensions). Similarly, BERT replaces ReLU with the GeLU activation function and adopts learnable positional embeddings instead of sinusoidal positional encoding. 

However, key differences include:

1. **Self-Attention Mechanism**: BERT employs bidirectional self-attention to capture contextual information from both directions, whereas GPT uses causal self-attention (masked to prevent future token visibility). In BERT, masks are applied only for padding.
2. **Tokenization**: BERT utilizes WordPiece tokenization, contrasting with GPT's Byte-Pair Encoding (BPE).
3. **Training Objectives**: BERT is pre-trained with Masked Language Modeling (MLM) and Next Sentence Prediction (NSP), while GPT and the original Transformer rely on standard autoregressive language modeling.
4. **Special Tokens**: During pre-training, BERT learns embeddings for [SEP]**, **[CLS], and sentence A/B embeddings. In GPT, [SEP] and [CLS] are introduced only during fine-tuning.

### Pre-training Tasks

#### Masked Language Model（MLM）

> Original Paper : 3.3.1 Task #1: Masked LM 

```
Input Sequence  : The man went to [MASK] store with [MASK] dog
Target Sequence :                  the                his
```

##### Rules:

Randomly 15% of input token will be changed into something, based on under sub-rules

1. Randomly 80% of tokens, gonna be a `[MASK]` token 
2. Randomly 10% of tokens, gonna be a `[RANDOM]` token(another word)
3. Randomly 10% of tokens, will be remain as same. But need to be predicted.

#### Next Sentence Prediction(NSP)

> Original Paper : 3.3.2 Task #2: Next Sentence Prediction

```
Input : [CLS] the man went to the store [SEP] he bought a gallon of milk [SEP]
Label : Is Next

Input = [CLS] the man heading to the store [SEP] penguin [MASK] are flight ##less birds [SEP]
Label = NotNext
```

This is for understanding the relationship between two text sentences, which is not directly captured by language modeling.

##### Rules:

1. Randomly 50% of next sentence, gonna be continuous sentence.
2. Randomly 50% of next sentence, gonna be unrelated sentence.



## [Pre-training](./pretrain.ipynb)

According to the settings described in the paper  [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) , BERT was pre-trained on BooksCorpus and Wikipedia datasets using the AdamW optimizer（$w = 0.01, \text{max-lr} = 1 \times 10^{-4}$）. During training, a linear warmup learning rate schedule was employed: the learning rate linearly increases over the first 10,000 steps, followed by a linear decay schedule.

## [Fine-tuning](./finetune.ipynb) 

After pre-training, BERT has acquired robust language understanding capabilities, which can be adapted to new tasks through fine-tuning. During fine-tuning, only minor architectural adjustments are required, such as adding task-specific classification heads for downstream tasks.

Since optimal hyperparameter values are task-specific, the original paper provided hyperparameter ranges for fine-tuning across different tasks:

- **Batch size**: 16, 32
- **Learning rate **：5e-5, 3e-5, 2e-5
- **Number of epochs**: 2, 3, 4

In this implementation, we use the AdamW optimizer（$w = 0.01, \text{max-lr} = 4 \times 10^{-5}$） with a batch size of 32 and 3 epochs. For learning rate scheduling, we adopt a linear warmup strategy: the learning rate linearly increases over the first 10,000 steps, followed by a linear decay schedule.

## Appendix

### 如何下载预训练好的BERT？

Run the following command in the terminal:

```bash
pip install -U huggingface-cli
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download bert-base-uncased --local-dir path/to/pretrained_dir
```

