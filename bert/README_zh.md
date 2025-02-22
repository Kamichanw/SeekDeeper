[📖English ReadMe](./README.md)

## Introduction

在这个BERT-base的实现中，我将展示如何在示例数据集上进行预训练；就微调部分，将展示从huggingface上加载google的预训练模型权重到自己实现的模型中，并在SST-2数据集上进行微调训练；最后给定文本由模型进行句子情感分类的推理。



## Model details

### Key differences with GPT

事实上，BERT的许多设计都是有意为之的，使其尽可能接近原本的GPT，从而使两种方法能够最小程度地进行比较。BERT_base的模型大小、注意力头数、层数都和GPT相同；同样的，BERT的激活函数从 ReLU 替换为了 GeLU，也使用 learnable positional embedding 而不是正余弦位置编码。

但具体来说还是有一定不同：

1.BERT使用的是原始Transformer 的Encoder，无需Decoder。也就是BERT是可以看到前后文信息的，而GPT是只能从历史上下文中获取信息。

2.BERT预训练的任务包括MLM和NSP，而GPT预训练的任务主要是预测给定文本序列中下一个词。

3.BERT在预训练的时候就学习到[SEP]、[CLS]和句子A/B的嵌入；GPT使用[SEP]和[CLS]，但它们只在微调的时候引入。

4.训练细节（包括数据集、学习率的设置）



### MLM 和 NSP

#### Masked Language Model（MLM）掩码语言模型

> Original Paper : 3.3.1 Task #1: Masked LM 

```
Input Sequence  : The man went to [MASK] store with [MASK] dog
Target Sequence :                  the                his
```

##### Rules:

输入的随机15%基于以下规则将会被改变为其他东西：

1. 随机80%的token—> `[MASK]`
2. 随机10%的token—>`[RANDOM]` token（其他的不一样的词）
3. 随机10%的token不变，但仍需要被预测。

#### Next Sentence Prediction(NSP) 下一句子预测

> Original Paper : 3.3.2 Task #2: Next Sentence Prediction

```
Input : [CLS] the man went to the store [SEP] he bought a gallon of milk [SEP]
Label : Is Next

Input = [CLS] the man heading to the store [SEP] penguin [MASK] are flight ##less birds [SEP]
Label = NotNext
```

用于理解两个文本句子之间的关系，这是大语言模型不能直接学习到的。

##### Rules:

1. 随机50%的句子下一句是连续的
2. 随机50%的句子下一句是不连续的



## [Pre-training](./pretrain.ipynb)

论文中的BERT是在BookCorpus和Wikipedia数据集上进行预训练的。在本实现中，我加载了BookCorpus和Wikipedia的第一个parquet 文件并将其合并成了一个**corpus.txt**文件。

语料的格式是同一行两个句子，句子间隐式地用 \t 分隔，示例如下：

```bash
Welcome to the \t the jungle\n
I can stay \t here all night\n
```

BERT使用WordPiece方法进行tokenize，首先对数据集tokenize生成词汇表；接着就是设定模型参数构建BERT模型，进行训练，具体流程参考pretrain.ipynb。预训练使用的优化器是Adam，其中lr为1e-4，β1和β2分别为0.9和0.999，L2 权重衰减为0.01。

## [Fine-tuning](./finetune.ipynb) 

在数据集上训练过后，BERT模型可以认为有学习到一定的语言能力。本实现选择在SST-2数据集上进行微调，用于文本情感分类任务。这首先需要在原本的BERT模型架构的最后加一个线性二分类层。我从huggingface上加载[bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)的权重到自己的模型中，然后在数据集上进行训练微调。使用的优化器是AdamW，其中lr为4e-5，β1和β2分别为默认的0.9和0.999，权重衰减同样为0.01。

## [Inferencing](./inference.ipynb) 

加载指定模型，对给定文本进行情感分类任务的推理。



## Reference

本实现参考如下：

1.[google-research/bert: TensorFlow code and pre-trained models for BERT](https://github.com/google-research/bert)

2.[codertimo/BERT-pytorch: Google AI 2018 BERT pytorch implementation](https://github.com/codertimo/BERT-pytorch)

3.[transformers/src/transformers/models/bert at 0de15c988b0d27758ce360adb2627e9ea99e91b3 · huggingface/transformers](https://github.com/huggingface/transformers/tree/0de15c988b0d27758ce360adb2627e9ea99e91b3/src/transformers/models/bert)