# From: https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import time
import torchvision.transforms as T
import re

import argparse, wandb

# Hyperparameters
EPOCHS = 5  # epoch
LR = 0.01 #5  # learning rate
BATCH_SIZE = 8  # batch size for training
EMBED_DIM = 64 # embedding size in model
MAX_LEN = 1024 # maximum text input length

# Get cpu, gpu device for training.
# mps does not (yet) support nn.EmbeddingBag.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

class CsvTextDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if idx >= self.__len__(): raise IndexError()
        text = self.data_frame.loc[idx, "article"]
        label = self.data_frame.loc[idx, "label_idx"]

        if self.transform:
            text = self.transform(text)

        return text, label

class SimpleTokenizer:
    def __call__(self, text):
        # Add a space between punctuation and words
        text = re.sub(r'([.,:;!?()])', r' \1 ', text)
        # Replace multiple whitespaces with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        # Tokenize by splitting on whitespace
        return text.split()

class Vocab:
    def __init__(self, oov_token, pad_token):
        self.idx2str = []
        self.str2idx = {}
        self.oov_index = 0
        self.add_tokens([oov_token, pad_token])
        self.oov_idx = self[oov_token]
        self.pad_idx = self[pad_token]

    def add_tokens(self, tokens):
        for token in tokens:
            if token not in self.str2idx:
                self.str2idx[token] = len(self.idx2str)
                self.idx2str.append(token)

    def __len__(self):
        return len(self.str2idx)

    def __getitem__(self, token):
        return self.str2idx.get(token, self.oov_index)

class CorpusInfo():
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.oov_token = '<OOV>' # out-of-vocabulary token
        self.pad_token = '<PAD>' # padding token
        
        self.vocab = Vocab(self.oov_token, self.pad_token)
        for text, _ in dataset:
            self.vocab.add_tokens(tokenizer(text))
        
        self.oov_idx = self.vocab[self.oov_token]
        self.pad_idx = self.vocab[self.pad_token]
        
        self.vocab_size = len(self.vocab)
        self.num_labels = len(set([label for (text, label) in dataset]))

class TextTransform():
    def __init__(self, tokenizer, vocab):
        self.tokenizer = tokenizer
        self.vocab = vocab

    def tokenize_and_numericalize(self, text):
        tokens = self.tokenizer(text)
        return [self.vocab[token] for token in tokens]

    def __call__(self, text):
        return self.tokenize_and_numericalize(text)
    
class MaxLen():
    def __init__(self, max_len):
        self.max_len = max_len
        
    def __call__(self, x):
        if len(x) > self.max_len:
            x = x[:self.max_len]
        return x
    
class PadSequence():
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    def __call__(self, batch):
        def to_int_tensor(x):
            return torch.from_numpy(np.array(x, dtype=np.int64))
        # Convert each sequence of tokens to a Tensor
        sequences = [to_int_tensor(x[0]) for x in batch]
        # Convert the full sequence of labels to a Tensor
        labels = to_int_tensor([x[1] for x in batch])
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.pad_idx)
        return sequences_padded, labels

def get_data():    
    train_data = CsvTextDataset(
        csv_file='./data/txt_train.csv',
        transform=None,
    )
    tokenizer = SimpleTokenizer()
    corpus_info = CorpusInfo(train_data, tokenizer)
    transform_txt = T.Compose([
        TextTransform(corpus_info.tokenizer, corpus_info.vocab),
        MaxLen(MAX_LEN),
    ])
    train_data = CsvTextDataset(
        csv_file='./data/txt_train.csv',
        transform=transform_txt,
    )
    val_data = CsvTextDataset(
        csv_file='./data/txt_val.csv',
        transform=transform_txt,
    )
    test_data = CsvTextDataset(
        csv_file='./data/txt_test.csv',
        transform=transform_txt,
    )

    # # IF YOU WANT TO PLOT THE HISTOGRAMS UNCOMMENT THIS PART
    #
    # import matplotlib.pyplot as plt
    # 
    # i = 0
    # plt.figure(figsize=(12,4))
    # for style, data, color in zip(["train", "validation", "test"], [train_data, val_data, test_data], ["lightblue", "purple", "darkgreen"]):
    #     i += 1
    #     if i == 3: i += 1
    #     plt.subplot((1 if i == 1 else 2),2,i)
    #     lengths = [len(text) for text, label in data]
    #     plt.hist(lengths, label=style, bins=20, color=color, histtype='bar')
    #     plt.legend()
    # plt.savefig("hist_lengths.png")
    # exit(0)


    collate_batch = PadSequence(corpus_info.pad_idx)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_batch)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_batch)

    for X, y in train_dataloader:
        print(f"Shape of X [B, N]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    
    return corpus_info, train_dataloader, val_dataloader, test_dataloader

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, text):
        embedded = self.embedding(text)
        return self.fc(embedded)

class LSTM_Model(nn.Module):
    """"
        Unidirectional LSTM model:

        (1) Embedding
        (2) LSTM (bidirectional)
             - hidden_size = 10
             - number of layers = 2
        (3) AdaptiveMaxPool1d
        (4) Linear(hidden_size, num_class)
    """
    def __init__(self, vocab_size, embed_dim, num_class):
        super(LSTM_Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=False)
        hidden_size = 10
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, 
            num_layers=2, batch_first=True
            #, dropout=0.25
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, text):
        embedded = self.embedding(text)   # Shape: (B, seq_length, embed_dim)
        out, _ = self.lstm(embedded)      # Shape: (B, seq_length, hidden_size)
        out = out.transpose(1, 2)
        x = self.pooling(out).squeeze(-1)
        x = self.fc(self.dropout(x))                    # Shape: (B, num_class)
        self.we_print = False
        return x

class LSTM_Model_bidirectional(nn.Module):
    """"
        Bidirectional LSTM model:

        (1) Embedding
        (2) LSTM (bidirectional)
             - hidden_size = 10
             - number of layers = 2
        (3) AdaptiveMaxPool1d
        (4) Linear(2*hidden_size, hidden_size)
        (5) ReLU()
        (6) Linear(hidden_size, num_class)
    """
    def __init__(self, vocab_size, embed_dim, num_class):
        super(LSTM_Model_bidirectional, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=False)
        hidden_size = 10
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, 
            num_layers=2, batch_first=True,
            bidirectional=True #, dropout=0.25
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(2*hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_class)

    def forward(self, text):
        embedded = self.embedding(text)   # Shape: (B, seq_length, embed_dim)
        out, _ = self.lstm(embedded)      # Shape: (B, seq_length, hidden_size * 2) -> x2 because of being "bidirectional"
        out = out.transpose(1, 2)
        #x = out[:, -1, :]                 # Shape: (B, hidden_size * 2)             -> get the output of the last time step for each sequence.
                                           #                                            only if we don't use the pooling layer
        x = self.pooling(out).squeeze(-1)
        #x = self.fc2(self.dropout(self.relu(self.fc(x))))    # UNCOMMENT TO TEST DROPOUT
        x = self.fc2(self.relu(self.fc(x)))                   # Shape: (B, num_class)
        self.we_print = False
        return x


def train_one_epoch(dataloader, model, criterion, optimizer, epoch):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 5
    start_time = time.time()

    for idx, (text, label) in enumerate(dataloader):
        text, label = text.to(device), label.to(device)
        optimizer.zero_grad()
        predicted_label = model(text)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch, idx, len(dataloader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(dataloader, model, criterion, dataname, epoch):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (text, label) in enumerate(dataloader):
            text, label = text.to(device), label.to(device)
            predicted_label = model(text)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    accuracy = total_acc / total_count
    
    # ADD LOGGINGS TO THE WANDB WEBSITE TO TRACK THE TRAINING

    wandb.log({
        f"{dataname}/accuracy": accuracy,
    }, step=epoch)
    
    return accuracy

def main(model_name, optim_name, run=0, table=None):
    corpus_info, train_dataloader, val_dataloader, test_dataloader = get_data()

    # CHOSE MODEL -> THIS CAN BE SET IN THE ARGPARSER

    model = (TextClassificationModel(corpus_info.vocab_size, EMBED_DIM, corpus_info.num_labels).to(device) if model_name == "base" else
                LSTM_Model(corpus_info.vocab_size, EMBED_DIM, corpus_info.num_labels).to(device) if model_name == "lstm" else
                    LSTM_Model_bidirectional(corpus_info.vocab_size, EMBED_DIM, corpus_info.num_labels).to(device))
    criterion = torch.nn.CrossEntropyLoss()

    # CHANGE THE OPTIMIZER TO ADAM -> THIS CAN BE SET IN THE ARGPARSER

    optimizer = (torch.optim.SGD(model.parameters(), lr=LR) if optim_name == "SGD" else
                                                    torch.optim.Adam(model.parameters(), lr=LR))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

    total_accu = None    
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train_one_epoch(train_dataloader, model, criterion, optimizer, epoch)
        accu_train = evaluate(train_dataloader, model, criterion, "train", epoch)
        accu_val = evaluate(val_dataloader, model, criterion, "validation", epoch)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print("-" * 59)
        print(
            "| end of epoch {:3d} | time: {:5.2f}s | "
            "valid accuracy {:8.3f} ".format(
                epoch, time.time() - epoch_start_time, accu_val
            )
        )
        print("-" * 59)

    # print("Checking the results of test dataset.")
    # acc_val = evaluate(val_dataloader, model, criterion)
    # acc_test = evaluate(test_dataloader, model, criterion)
    # print("test accuracy {:8.3f}".format(acc_test))

    
    # # UNCOMMENT TO GET THE DATA FOR THE TABLES OF THE THREE RUNS
    # table.add_data(run, acc_val, acc_test)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Text Classifier')
    parser.add_argument('--model_name', default="base", choices=["base", "lstm", "lstm_2dir"], help='Name of the model to train', type=str)
    parser.add_argument('--optim_name', default="SGD", choices=["SGD", "ADAM"], help='Name of the optimizer', type=str)
    parser.add_argument('--run_name', default="Test", help='Name for the wandb run', type=str)
    args = parser.parse_args()
    
    wandb.init(
        project="HW0-Text-Classification",
        name=args.run_name
    )
    
    main(args.model_name, args.optim_name)
    
    # # IF YOU WANT TO CREATE A TABLE OF THE ACCURACY OF THREE RUNS UNCOMMENT THIS PART  
    # 
    # accuracy_table = wandb.Table(columns = ["Run", "Valition", "Test"])
    # 
    # for run in range(1, 4):
    #     main(args.model_name, args.optim_name, run, accuracy_table)
    # 
    # wandb.log({"accuracy_table": accuracy_table})

    wandb.finish()