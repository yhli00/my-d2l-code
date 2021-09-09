# GRU的简洁实现，实现字符级别的语言模型，数据由随机采样生成
# num_epochs = 500
# num_steps = 35
# lr = 1
# max_norm = 1.0
# 在长度为10000的语料库中训练，ppl=1.379


import math
import re
import torch
from torch import nn
from collections import Counter
import random
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


torch.manual_seed(12345)
file_path = '../data/timemachine.txt'
device = torch.device('cuda:2')


def read_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return [re.sub('[^a-zA-Z]+', ' ', line).strip().lower() for line in lines]


def tokenize(lines, token_type='char'):
    if token_type == 'word':
        return [line.split() for line in lines]
    elif token_type == 'char':
        return [list(line) for line in lines]


class Vocab:
    def __init__(self, tokens, min_freqs=-1):
        self.corpus = [token for line in tokens for token in line]
        counter = Counter(self.corpus)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.unk = 0
        self.token_to_idx = {'<unk>': 0}
        self.idx_to_token = ['<unk>']
        for t, f in self.token_freqs:
            if f >= min_freqs:
                self.idx_to_token.append(t)
                self.token_to_idx[t] = len(self.idx_to_token) - 1
    
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, list):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        return [self.idx_to_token[indic] for indic in indices]


def data_iter_random(corpus, batch_size, num_steps):
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_seqs = (len(corpus) - 1) // num_steps
    initial_idx = list(range(0, num_seqs * num_steps, num_steps))
    random.shuffle(initial_idx)
    num_batches = num_seqs // batch_size
    for i in range(0, num_batches * batch_size, batch_size):
        indices = initial_idx[i: i + batch_size]
        X = [corpus[j: j + num_steps] for j in indices]
        Y = [corpus[j + 1: j + 1 + num_steps] for j in indices]
        yield torch.tensor(X), torch.tensor(Y)


class data_iter:
    def __init__(self, corpus, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.corpus = corpus
    
    def __iter__(self):
        return data_iter_random(self.corpus, self.batch_size, self.num_steps)


class GRUModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, batch_size):
        super().__init__()
        self.gru = nn.GRU(input_size=vocab_size, hidden_size=hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
    
    def get_init_state(self):
        return torch.zeros((1, self.batch_size, self.hidden_size), device=device)
    
    def forward(self, x, state):  # x:[L, B, V] state:[1, B, H]
        O, _ = self.gru(x, state)  # O:[L, B, H] _:[1, B, H]
        return self.output(O.reshape(-1, O.shape[-1]))


def train_epoch(net, data_iter, loss_fn, max_norm, optimizer):
    total_loss = 0.
    total_num = 0.
    for X, Y in data_iter:
        X = F.one_hot(X.T, net.vocab_size).float().to(device)
        Y = Y.T.reshape(-1).long().to(device)
        y_hat = net.forward(X, net.get_init_state())
        loss = loss_fn(y_hat, Y)
        total_loss += loss.detach().clone().item() * Y.numel()
        total_num += Y.numel()
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(net.parameters(), max_norm)
        optimizer.step()
    return math.exp(total_loss / total_num)


def train(net, num_epochs, data_iter, lr, max_norm, print_every):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        ppl = train_epoch(net, data_iter, loss_fn, max_norm, optimizer)
        if (epoch + 1) % print_every == 0:
            print(f'epoch {epoch + 1}, ppl {ppl:.6f}')


if __name__ == '__main__':
    lr = 1
    max_norm = 1.0
    num_epochs = 500
    num_steps = 35
    batch_size = 32
    hidden_size = 256
    lines = read_file(file_path)
    tokens = tokenize(lines, token_type='char')
    vocab = Vocab(tokens)
    corpus = [token for line in tokens for token in line]
    corpus = corpus[:10000]
    corpus = vocab[corpus]
    seq_data_iter = data_iter(corpus, batch_size, num_steps)
    net = GRUModel(len(vocab), hidden_size, batch_size)
    net = net.to(device)
    train(net, num_epochs, seq_data_iter, lr, max_norm, print_every=1)

