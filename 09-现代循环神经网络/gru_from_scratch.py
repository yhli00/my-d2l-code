# GRU从零开始实现，实现字符级别的语言模型，数据由随机采样生成
# num_epochs = 500
# num_steps = 35
# lr = 1
# max_norm = 1.0
# 在长度为10000的语料库中训练，ppl=1.467


import math
import re
import torch
from collections import Counter
import random
import torch.nn.functional as F


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


class GRUModel:
    def __init__(self, vocab_size, hidden_size, batch_size, num_steps):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.params = self._getParams()
        
    
    def _getParams(self):
        W_xz, W_hz, b_z = self._getThreePrams()  # 更新门
        W_xr, W_hr, b_r = self._getThreePrams()  # 重置门
        W_xh, W_hh, b_h = self._getThreePrams()  # 计算H_candidate
        W_hq = torch.normal(size=(self.hidden_size, self.vocab_size), mean=0., std=0.01, device=device, requires_grad=True)
        b_q = torch.zeros((self.batch_size, self.vocab_size), device=device, requires_grad=True)
        return [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]

    def _getThreePrams(self):
        W_xh = torch.normal(size=(self.vocab_size, self.hidden_size), mean=0., std=0.01, device=device, requires_grad=True)
        W_hh = torch.normal(size=(self.hidden_size, self.hidden_size), mean=0., std=0.01, device=device, requires_grad=True)
        b = torch.zeros((self.batch_size, self.hidden_size), device=device, requires_grad=True)
        return [W_xh, W_hh, b]
    
    def forward(self, x, state):  # x:[L, B, V], state:[B, H]
        outputs = []
        H = state
        W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = self.params
        for t in x:
            Z = torch.sigmoid(t @ W_xz + H @ W_hz + b_z)  # [B, H]
            R = torch.sigmoid(t @ W_xr + H @ W_hr + b_r)  # [B, H]
            H_candidate = torch.tanh(t @ W_xh + (R * H) @ W_hh + b_h)  # [B, H]
            H = Z * H + (1 - Z) * H_candidate
            Y = H @ W_hq + b_q
            outputs.append(Y)  # [L, B, V]
        return torch.cat(outputs, dim=0), H
    
    def get_init_state(self):
        return torch.zeros((self.batch_size, self.hidden_size), dtype=torch.float, device=device)


def sgd(params, lr):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad
            param.grad.zero_()


def grad_clipping(params, max_norm):
    with torch.no_grad():
        norm = torch.sqrt(torch.sum(torch.tensor([(p.grad ** 2).sum() for p in params])))
        if norm > max_norm:
            for param in params:
                param.grad *= (max_norm / norm)


def train_epoch(net, data_iter, loss_fn, lr, max_norm):
    total_loss = 0.
    total_num = 0.
    for X, Y in data_iter:
        X = F.one_hot(X.T, net.vocab_size).float().to(device)
        Y = Y.T.reshape(-1).long().to(device)
        y_hat, _ = net.forward(X, net.get_init_state())
        loss = loss_fn(y_hat, Y)
        total_loss += loss.detach().clone().item() * Y.numel()
        total_num += Y.numel()
        loss.backward()
        grad_clipping(net.params, max_norm)
        sgd(net.params, lr)
    return math.exp(total_loss / total_num)


def train(net, num_epochs, data_iter, lr, max_norm, print_every):
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        ppl = train_epoch(net, data_iter, loss_fn, lr, max_norm)
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
    net = GRUModel(len(vocab), hidden_size, batch_size, num_steps)
    train(net, num_epochs, seq_data_iter, lr, max_norm, print_every=1)

