[📖中文 ReadMe](./README_zh.md)
## Introduction

## Model details

### Key differences with GPT

In fact, many of BERT's design choices are intentional, aiming to make it as similar as possible to the original GPT, so that the two methods can be compared with minimal differences. The model size, number of attention heads, and number of layers in BERT_base are the same as GPT; similarly, BERT uses the GeLU activation function instead of ReLU, and uses learnable positional embeddings rather than sine-cosine position encodings.

However, there are still some differences:

1. BERT uses the original Transformer Encoder and does not require a Decoder. This means that BERT can access both the previous and future context, while GPT can only access the historical context.
2. BERT's pretraining tasks include MLM (Masked Language Model) and NSP (Next Sentence Prediction), while GPT's pretraining task mainly involves predicting the next word in a given text sequence.
3. During pretraining, BERT learns embeddings for [SEP], [CLS], and sentence A/B; GPT uses [SEP] and [CLS], but they are only introduced during fine-tuning.
4. Training details (including datasets and learning rate settings).



### MLM and NSP

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

"Is this sentence can be continuously connected?"

understanding the relationship, between two text sentences, which is not directly captured by language modeling

##### Rules:

1. Randomly 50% of next sentence, gonna be continuous sentence.
2. Randomly 50% of next sentence, gonna be unrelated sentence.



## [Pre-training](./pretrain.ipynb)

The BERT model in the paper is pretrained on the BookCorpus and Wikipedia datasets. In this implementation, I load the first parquet file of BookCorpus and Wikipedia and convert them into one txt file as **corpus.txt**. 

The format of the corpus consists of two sentences on the same line, implicitly separated by a tab (`\t`), as shown in the example below:

```
Welcome to the \t the jungle\n
I can stay \t here all night\n
```

BERT uses the WordPiece method for tokenization, first generating a vocabulary by tokenizing the dataset; then, the model parameters are set and the BERT model is constructed for training. For the specific process, refer to pretrain.ipynb. The optimizer used for pretraining is Adam, with a learning rate (lr) of 1e-4, β1 and β2 set to 0.9 and 0.999, respectively, and an L2 weight decay of 0.01.	

## [Fine-tuning](./finetune.ipynb) 

After training on the dataset, the BERT model can be considered to have learned some language abilities. In this implementation, fine-tuning is performed on the SST-2 dataset for a text sentiment classification task. This requires adding a linear binary classification layer to the end of the original BERT model architecture. I load the weights of [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) from Hugging Face into my model, then fine-tune it on the dataset. The optimizer used is AdamW, with a learning rate (lr) of 2e-5, β1 and β2 set to the default values of 0.9 and 0.999, and the weight decay is also 0.01.

## [Inferencing](./inference.ipynb) 

Load the specified model and perform inference for the sentiment classification task on a given text.



## Reference

This implementation is based on the following:

1.[google-research/bert: TensorFlow code and pre-trained models for BERT](https://github.com/google-research/bert)

2.[codertimo/BERT-pytorch: Google AI 2018 BERT pytorch implementation](https://github.com/codertimo/BERT-pytorch)

3.[transformers/src/transformers/models/bert at 0de15c988b0d27758ce360adb2627e9ea99e91b3 · huggingface/transformers](https://github.com/huggingface/transformers/tree/0de15c988b0d27758ce360adb2627e9ea99e91b3/src/transformers/models/bert)