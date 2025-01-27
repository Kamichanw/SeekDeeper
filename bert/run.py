from torch.utils.data import DataLoader

import config
from data import WordVocab, BERTDataset
from modules.bert import BERT
from modules.pretrain import BERTTrainer


def train():

    print("Loading Vocab", config.vocab_path)
    vocab = WordVocab.load_vocab(config.vocab_path)
    print("Vocab Size: ", len(vocab))

    print("Loading Train Dataset", config.train_dataset)
    train_dataset = BERTDataset(config.train_dataset, vocab, seq_len=config.sequence_length,
                                corpus_lines=config.corpus_lines, on_memory=config.on_memory)
    print("Train Dataset Size: ", len(train_dataset))

    print("Loading Test Dataset", config.test_dataset)
    test_dataset = BERTDataset(config.test_dataset, vocab, seq_len=config.sequence_length, on_memory=config.on_memory) \
        if config.test_dataset is not None else None

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers) \
        if test_dataset is not None else None

    print("Building BERT model")
    bert = BERT(len(vocab), hidden=config.hidden_size, n_layers=config.num_layers, attn_heads=config.attention_heads)

    print("Creating BERT Trainer")
    trainer = BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                          lr=config.learning_rate, betas=(config.adam_beta1, config.adam_beta2), weight_decay=config.adam_weight_decay,
                          with_cuda=config.with_cuda, cuda_devices=config.cuda_devices, log_freq=config.log_freq)

    print("Training Start")
    for epoch in range(config.epochs):
        trainer.train(epoch)
        trainer.save(epoch, config.output_path)

        if test_data_loader is not None:
            trainer.test(epoch)


if __name__ == "__main__":
    train()