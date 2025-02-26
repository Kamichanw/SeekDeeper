[📖 English ReadMe](./README.md)

## Introduction
在这个 BERT 实现中，我们将展示如何在 [BookCorpus](https://huggingface.co/datasets/bookcorpus/bookcorpus) 和 [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) 数据集上进行预训练，然后加载 Hugging Face 提供的官方预训练权重，并在 [Stanford Sentiment Treebank (SST-2)](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf) 数据集上进行微调，复现论文中的效果。

## Model Details

### Key Differences with Vanilla Transformer and GPT

BERT的许多设计都是有意为之的，使其尽可能接近原本的GPT，从而使两种方法能够最小程度地进行比较。BERT_base的模型大小、注意力头数、层数都和GPT相同；同样的，BERT的激活函数从 ReLU 替换为了 GeLU，也使用 learnable positional embedding 而不是正余弦位置编码。但具体来说还是有不同之处：

1. **BERT 的自注意力机制**：BERT 通过双向自注意力来捕捉上下文信息，而 GPT 仅使用因果自注意力。换而言之，BERT 的 mask 只用于 padding。
2. **BERT 的词嵌入**：BERT 使用的是 WordPiece 的分词方式，这与 GPT 的 Byte-Pair Encoding（BPE）不同。
3. **训练目标**：BERT 使用了 Masked Language Modeling (MLM) 和 Next Sentence Prediction (NSP) 作为预训练目标，而 GPT 和原始 Transformer 采用的是传统的语言建模目标。
4. **token处理**：BERT在预训练的时候就学习到[SEP]、[CLS]和句子A/B的嵌入；GPT使用[SEP]和[CLS]，但它们只在微调的时候引入。

### Pre-training Tasks

#### Masked Language Modeling (MLM)

> Original Paper : 3.3.1 Task #1: Masked LM 

```
Input Sequence  : The man went to [MASK] store with [MASK] dog
Target Sequence :                  the                his
```

##### Rules:

输入的随机15%基于以下规则将会被改变为其他东西：

1. 随机80%的token -> `[MASK]`
2. 随机10%的token -> `[RANDOM]` token（其他的不一样的词）
3. 随机10%的token不变，但仍需要被预测。

#### Next Sentence Prediction (NSP)

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
根据 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) 论文中的设置，BERT 在 BooksCorpus 和Wikipedia 数据集上进行预训练，使用 AdamW 优化器（$w = 0.01, \text{max-lr} = 1 \times 10^{-4}$）。训练时使用了线性增长的学习率策略，学习率在前 10000 步线性增加，之后使用线性衰减策略调整学习率。

## [Fine-tuning](./finetune.ipynb)
预训练完成后，BERT 已经获得了较强的语言理解能力，可以通过微调来适应新的任务。在微调时，只需要对模型结构做轻微调整，并在下游任务中添加适当的分类头。

由于最佳超参数值取决于具体任务，原论文针对不同任务的微调给出了超参数范围：

- **Batch size**：16，32
- **Learning rate **：5e-5, 3e-5, 2e-5
- **Number of epochs**: 2, 3, 4

在本实现中，我们使用了 AdamW 优化器（$w = 0.01, \text{max-lr} = 4 \times 10^{-5}$），并选用 batch size 为 32 ，epoch 数为 3 。微调同样使用了线性增长的学习率策略，学习率在前 10000 步线性增加，之后使用线性衰减策略调整学习率。

## Appendix
### How to Download Pretrained BERT?
在命令行运行以下指令：
```bash
pip install -U huggingface-cli
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download bert-base-uncased --local-dir path/to/pretrained_dir
```
