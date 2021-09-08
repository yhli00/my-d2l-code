# pytorch实现基于单向rnn的字符级语言模型
# 数据集使用随机采样
# lr=1
# SGD优化器
# num_steps=35
# num_epochs=500
# ppl=1.614


import math
import torch
from torch import nn
from torch import optim
from torch.nn.utils import clip_grad_norm_
import re
from collections import Counter
import random
import torch.nn.functional as F


file_name = '../data/timemachine.txt'
device = torch.device('cuda:3')
torch.manual_seed(12345)


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
    
    def to_token(self, indices):
        return [self.idx_to_token[indic] for indic in indices]


def seq_data_iter_random(corpus, batch_size, num_steps):
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_seqs = (len(corpus) - 1) // num_steps
    initial_idx = list(range(0, num_seqs * num_steps, num_steps))
    random.shuffle(initial_idx)
    num_batches = num_seqs // batch_size
    for i in range(0, num_batches * batch_size, batch_size):
        indices = initial_idx[i: batch_size + i]
        Xs = [corpus[idx: idx + num_steps] for idx in indices]
        Ys = [corpus[idx + 1: idx + 1 + num_steps] for idx in indices]
        yield torch.tensor(Xs), torch.tensor(Ys)


class data_iter:
    def __init__(self, corpus, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.corpus = corpus
        self.iter_fn = seq_data_iter_random
    
    def __iter__(self):
        return self.iter_fn(self.corpus, self.batch_size, self.num_steps)


class RNNModel(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.rnn = nn.RNN(input_size=vocab_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, state):  # x:[L, B, V], state:[1, B, H]
        output, _ = self.rnn(x, state)  # output:[L, B, H]
        return self.lin(output.reshape(-1, output.shape[-1]))

    
    def getInitState(self, batch_size):
        return torch.zeros((1, batch_size, self.hidden_size), device=device)


def train_epoch(net, data_iter, batch_size, optimizer, loss_fn, max_norm):
    net.train()
    total_loss = 0.
    total_num = 0.
    for x, y in data_iter:
        x = F.one_hot(x.T, net.vocab_size).float().to(device)
        y = y.T.reshape(-1).long().to(device)  # [L*B]
        y_hat = net(x, net.getInitState(batch_size))  # [L*B, V]
        loss = loss_fn(y_hat, y)
        total_loss += loss.detach().clone().item() * y.numel()
        total_num += y.numel()
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(net.parameters(), max_norm)
        optimizer.step()
    return math.exp(total_loss / total_num)


def train(net, num_epochs, lr, data_iter, batch_size, max_norm, print_every):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        ppl = train_epoch(net, data_iter, batch_size, optimizer, loss_fn, max_norm)
        if (epoch + 1) % print_every == 0:
            print(f'epoch {epoch + 1}, ppl {ppl:.8f}')



if __name__ == '__main__':
    hidden_size = 256
    batch_size = 32
    num_epochs = 500
    num_steps = 35
    lr = 1
    lines = read_file(file_name)
    tokens = tokenize(lines)
    vocab = Vocab(tokens)
    corpus = [token for line in tokens for token in line]
    corpus = vocab[corpus]
    corpus = corpus[:10000]
    random_data_iter = data_iter(corpus, batch_size, num_steps)
    net = RNNModel(hidden_size, len(vocab))
    net = net.to(device)
    train(net, num_epochs, lr, random_data_iter, batch_size, max_norm=1., print_every=1)

