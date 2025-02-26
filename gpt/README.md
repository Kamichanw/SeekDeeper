[📖中文 ReadMe](./README_zh.md)
## Introduction  
In this GPT implementation, I will demonstrate how to pre-train on the [BookCorpus](https://huggingface.co/datasets/bookcorpus/bookcorpus) dataset, then load the official pre-trained weights from [huggingface](https://huggingface.co/openai-community/openai-gpt) into our model, fine-tune on the [Stanford Sentiment Treebank (SST-2)](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf) dataset, and replicate the results mentioned in the paper.

## Model details  
### Key differences with vanilla transformer  
1. GPT only uses the Decoder from the original transformer. Since the Encoder is no longer needed, the cross-attention with the Encoder is removed, and only the causal self-attention layer is used.  
2. The feed-forward neural network layer in GPT's Decoder has one less dropout. Additionally, the activation function is replaced from ReLU to GeLU.  
3. GPT slightly increases the model size and fixes the vocabulary size (since GPT is a pre-trained model).  
4. GPT uses [Weight Tying](https://arxiv.org/abs/1608.05859), sharing the weights between the language modeling head and the token embedding layer.  
5. GPT uses learnbale positional embedding instead of sinusoidal positional encoding.

The summary is shown in the table below:  
<table>  
  <thead>  
    <tr style="font-weight: bold; border-bottom: 2px solid">  
      <th></th>  
      <th style="text-align: center">GPT</th>  
      <th style="text-align: center">Transformer</th>  
    </tr>  
  </thead>  
  <tbody style="text-align:center">  
    <tr>  
      <td>Positional encoding</td>  
      <td> learnable </td>  
      <td> sinusoidal(mainstream) </td>  
    </tr>  
    <tr>  
      <td>num_attention_heads</td>  
      <td>12</td>  
      <td>8</td>  
    </tr>  
    <tr>  
      <td>num_hidden_layers</td>  
      <td>12</td>  
      <td>6 encoder layers, 6 decoder layers</td>  
    </tr>  
    <tr>  
      <td>hidden_size</td>  
      <td>768</td>  
      <td>512</td>  
    </tr>  
    <tr>  
      <td>vocab_size</td>  
      <td>40478</td>  
      <td>depends on dataset</td>  
    </tr>  
    <tr>  
      <td>FFN path</td>  
      <td style="text-align:left">  
      <pre>  
      <code>  
      mlpf = lambda x: dropout(fc2(gelu(fc1(x))))  
      x = x + layer_norm(mlpf(x))  
      </code>  
      </pre>  
      </td>  
      <td style="text-align:left">  
      <pre>  
      <code>  
      mlpf = lambda x: dropout(fc2(dropout(relu(fc1(x)))))  
      x = x + layer_norm(mlpf(x))  
      </code>  
      </pre>  
      </td>  
    </tr>  
  </tbody>  
</table>

### [Byte-pair encoding (BPE)](./modules/bpe.py)  
BPE is a tokenization method where the core idea is to merge the most frequently occurring character pairs to construct larger subword units, thereby reducing the vocabulary size and handling the problem of rare words. It requires training on a corpus to obtain a vocabulary before encoding and decoding can take place.

Since the [BookCorpus](https://huggingface.co/datasets/bookcorpus/bookcorpus) dataset provided by Huggingface has been meticulously post-processed, we cannot fully replicate the results of the [original GPT code](https://github.com/openai/finetune-transformer-lm). I have only implemented the encoding and decoding parts based on the original implementation in [text_utils.py](https://github.com/openai/finetune-transformer-lm/blob/master/text_utils.py). If you're interested in the BPE training process, you can refer to Karpathy's [minbpe](https://github.com/karpathy/minbpe).

#### Training  
1. Pre-tokenization: Use [`ftfy`](https://github.com/rspeer/python-ftfy) to normalize Unicode characters, unify non-standard punctuation, replace all whitespace characters with `\n`, and then tokenize using spacy's [en_core_web_sm](https://spacy.io/models/en#en_core_web_sm) model (see [bpe.py](./modules/bpe.py)).  
2. Initialize the vocabulary: Split the entire text corpus into single-character subword units, appending `</w>` to the last character. As shown in the trained vocabulary file [encoder_bpe_40000.json](./datasets/bookcorpus/encoder_bpe_40000.json), ids 1-238 are single characters, and 239-476 are in the form of a single character + `</w>`. The `</w>` here represents the end of a token. For example, in the word `bamboo`, the last `o` is treated as `o</w>` to distinguish it from the second-to-last `o`.  
3. Count the frequency of bi-gram character pairs.  
4. Merge the most frequently occurring character pairs and form a new subword unit. Update the vocabulary in the corpus and record this merge operation.  
5. Repeat steps 3-4 for 40000 times, creating 40000 new subword units based on the 476 single characters. Adding `<unk>` and `\n</w>` results in a total of 40478 tokens.

#### Encoding  
0. Load the trained vocabulary.  
1. Pre-tokenization: Pre-tokenize the input text, same as in the training phase.  
2. Split each subword into single-character subword units, appending `</w>` to the last character.  
3. Count the frequency of bi-gram character pairs.  
4. Choose the character pair that was merged the earliest in the vocabulary, forming a new subword unit. Replace the current text with the new subword unit.  
5. Repeat steps 3-4 until no more valid bi-grams exist or only one character unit remains.  
6. Cache the result and map the subword units to the corresponding token ids in the vocabulary.

#### Decoding  
0. Load the trained vocabulary.  
1. Create a reverse mapping from the vocabulary and map the given token ids back to the original subwords.
  
## [Pre-training](./pretrain.ipynb)  
Based on the setup in [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) Sec. 4.1, we use AdamW ($w = 0.01, \text{max-lr} = 2.4\times 10^{-4}$) as the optimizer on the BooksCorpus dataset. Note: **no weight decay is applied to the parameters in bias and scaling layers (`LayerNorm`)**. The learning rate is linearly increased from 0 over 2000 steps to $\text{max-lr}$, followed by a cosine annealing learning rate schedule. The model is trained for 100 epochs on randomly sampled minibatches of 64 continuous sequences of 512 tokens each.

To sample the specified number of continuous token sequences, we need to first convert the text in the original dataset to token ids using bpe. However, our custom tokenization implementation is very slow, so experiments are conducted on a small portion of the dataset. If you wish to try more data, you can modify the `loading_ratio` parameter in `load_data` within [pretrain.ipynb](./pretrain.ipynb).

## [Fine-tuning](./finetune.ipynb)  
After training on the BookCorpus dataset, GPT has gained a certain level of language ability. To apply it to a new dataset, only minor adjustments to the model structure and input are needed.

<div>  
  <img src="./images/gpt-train.png" alt="GPT architecture and training objectives used in other works" style="width: 100%; height: auto;">  
</div>

In Sec. 3.2 of the original paper, it is mentioned that adding a language modeling loss as an auxiliary objective during fine-tuning helps learning because (a) it improves the generalization ability of the supervised model, and (b) it accelerates convergence. Therefore, in addition to adding new tokens (`<start>`, and `<extract>`) to the vocabulary, the output of the decoder backbone needs to be fed into a newly added classification head.

Fine-tuning generally reuses the hyperparameter settings from pre-training. A dropout layer ($p = 0.1$) is added before the classifier. The learning rate is set to $6.25e^{-5}$, and the batch size is set to 32. In most cases, training for 3 epochs is sufficient. Additionally, a linear learning rate decay strategy with warmup is used, with a warmup proportion of $0.2\%$ of the total training steps. The weight of the classification loss is set to 0.5.

## Appendix  
### How to download pretrained GPT?  
Run the following command in the terminal  
```bash  
pip install -U huggingface-cli  
export HF_ENDPOINT=https://hf-mirror.com  
huggingface-cli download openai-community/openai-gpt --local-dir path/to/pretrained_dir  
```