# SeekDeeper: 最小化的流行人工智能模型实现
[\[📖English ReadMe\]](./README.md)

## 动机
官方的代码库往往考虑了太多工程化的细节，对于初学者来说可能会感到负担过重。这个仓库旨在基于PyTorch使用尽可能少的代码实现各种模型，使学习者更容易理解和复现结果。此外，大部分的教程都缺乏一套完整的流程讲解，即只含有模型的内容而忽视了数据的加载、训练，这同样导致初学者走马观花，难以动手实操。

## 模型

<table>
  <thead>
    <tr style="font-weight: bold; border-bottom: 2px solid">
      <th>Model</th>
      <th>Paper</th>
      <th>Official or Reference Repository</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Transformer</td>
      <td><a href="https://arxiv.org/abs/1706.03762">Attention Is All You Need</a></td>
      <td><a href="https://github.com/hyunwoongko/transformer">https://github.com/hyunwoongko/transformer</a></td>
    </tr>
    <tr>
      <td>GPT</td>
      <td><a href="https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf">Improving Language Understanding by Generative Pre-Training</a></td>
      <td rowspan=2><a href="https://github.com/openai/finetune-transformer-lm">https://github.com/openai/finetune-transformer-lm</a></br><a href="https://github.com/openai/gpt-2">https://github.com/openai/gpt-2</a></br><a href="https://github.com/karpathy/nanoGPT">https://github.com/karpathy/nanoGPT</a></br><a href="https://github.com/karpathy/minGPT">https://github.com/karpathy/minGPT</a></td>
    </tr>
    <tr>
      <td>GPT-2</td>
      <td><a href="https://cdn.openai.com/research-covers/language_models_are_unsupervised_multitask_learners.pdf">Language Models are Unsupervised Multitask Learners</a></td>
    </tr>
    <tr>
      <td>ViT</td>
      <td><a href="https://arxiv.org/pdf/2010.11929">An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale</a></td>
      <td><a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py">Huggingface ViT implementation</a></td>
    </tr>
    <tr>
      <td>GAN</td>
      <td><a href="https://arxiv.org/abs/1406.2661">Generative Adversarial Networks</a></td>
      <td><a href="https://github.com/goodfeli/adversarial">https://github.com/goodfeli/adversarial</a></td>
    </tr>
    <tr>
      <td>DCGAN</td>
      <td><a href="https://arxiv.org/pdf/1511.06434">Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks</a></td>
      <td><a href="https://github.com/Newmu/dcgan_code">https://github.com/Newmu/dcgan_code</a></td>
    </tr>
    <tr>
      <td>WGAN-GP</td>
      <td><a href="https://arxiv.org/pdf/1704.00028">Improved Training of Wasserstein GANs</a></td>
      <td><a href="https://github.com/igul222/improved_wgan_training">https://github.com/igul222/improved_wgan_training</a></td>
    </tr>
  </tbody>
</table>


## 目录结构

对于每个模型而言，其典型目录结构如下：

```
<model name>/
├── checkpoints/
├── modules/
├── datasets/
├── images/
├── README.md
├── data.py
├── config.py
├── train.ipynb
└── inference.ipynb
```

- **checkpoints/**：包含已训练好的模型权重，以供`inference.ipynb`中载入直接使用。有时也会直接从官方代码库中载入预训练参数。
- **components/**：包含实现模型所需的组件。
- **datasets/**：包含训练或推理验证时所需的数据集，有时可能会通过代码下载到该目录中。
- **images/**：包含用于文档的示例图片。
- **README.md**：介绍实现的任务，并描述实现的细节。
- **data.py**：定义了`Dataset`、`Dataloader`或预处理数据。
- **config.py**：定义了实验所需的超参数。
- **train.ipynb**：以清晰的方式展现从数据加载、预处理，到训练、评估的一系列过程。
- **inference.ipynb**：加载`checkpoints/`目录下的模型参数并进行推理。

## 许可协议

本项目使用MIT许可证。
