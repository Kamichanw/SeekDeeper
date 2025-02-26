# SeekDeeper: Minimal Implementations of Popular AI Models
[\[📖中文ReadMe\]](./README_zh.md)

## Motivation
Official code repositories often include many engineering details, which can be overwhelming for beginners. This repository aims to implement various models with as little code as possible using PyTorch, making it easier for learners to understand and reproduce results. Additionally, most tutorials lack a complete workflow, focusing only on the model without considering data loading and training. This makes it difficult for beginners to apply their knowledge in practice.

## Models
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
      <td>Bert</td>
      <td><a href="https://arxiv.org/pdf/1810.04805">BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</a></td>
      <td><a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py">Huggingface Bert implementation</a></td>
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


## Directory Structure

For each model, the typical directory structure is as follows:

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

- **checkpoints/**: Contains pre-trained model weights for direct use in `inference.ipynb`. Sometimes, pre-trained parameters from official repositories are loaded directly.
- **modules/**: Contains modules necessary for model implementation.
- **datasets/**: Contains datasets required for training or inference validation, which may sometimes be downloaded to this directory via code.
- **images/**: Contains images for README.md of this model.
- **README.md**: Introduces the implemented task and describes the implementation details.
- **data.py**: Defines `Dataset`, `Dataloader`, or data preprocessing.
- **config.py**: Defines hyperparameters needed for the experiment.
- **train.ipynb**: Clearly presents the process from data loading, preprocessing, to training and evaluation.
- **inference.ipynb**: Loads model parameters from the `checkpoints/` directory for inference.

## License

This project uses the MIT License.
