# pytorch实现基于rnn的字符级别语言模型
# 1、实现文件读取函数
# 2、实现分词函数
# 3、实现词表类
# 4、实现获得语料和词表的函数
# 5、实现随机采样函数
# 6、从零实现rnn:
#     + 实现优化器sgd
#     + 实现模型(模型的输入为one-hot向量)
#     + 实现预测函数（预测时先预热得到隐藏状态）
# 7、实现梯度裁剪
# 8、实现训练函数
# lr=1,num_epochs=500
# num_steps=35
# hidden_size=512
# 使用语料库中的10000个token，num_epoch=500:ppl=1.330
# 使用整个语料库（170580个token），num_epoch=500:ppl=3.778


import torch
import re
import random
import torch.nn.functional as F
from collections import Counter
from torch import nn
import math


torch.manual_seed(12345)
filename = '../data/timemachine.txt'


def read_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return [re.sub('[^a-zA-Z]+', ' ', line).lower().strip() for line in lines]


def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]


class Vocab:
    def __init__(self, tokens, min_freqs=0):
        self.corpus = [token for line in tokens for token in line]
        counter = Counter(self.corpus)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.unk = 0
        self.token_to_idx = {'<unk>': 0}
        self.idx_to_token = ['<unk>']
        for x, y in self.token_freqs:
            if y >= min_freqs:
                self.idx_to_token.append(x)
                self.token_to_idx[x] = len(self.idx_to_token) - 1
        
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, list):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, token_indices):
        return [self.idx_to_token[indic] for indic in token_indices]


class seq_data_sample:
    def __init__(self, corpus, batch_size, num_steps, is_random=False):
        self.corpus = corpus
        self.batch_size = batch_size
        self.num_steps = num_steps
        if is_random:
            self.seq_data_sample_fn = self._seq_data_iter_random
        else:
            self.seq_data_sample_fn = self._seq_data_iter_sequential
        
    # 随机采样
    def _seq_data_iter_random(self):
        batch_size = self.batch_size
        num_steps = self.num_steps
        self.corpus = self.corpus[random.randint(0, num_steps - 1):]
        num_seqs = (len(self.corpus) - 1) // num_steps
        num_batches = num_seqs // batch_size
        initial_indices = list(range(0, num_seqs * num_steps, num_steps))
        random.shuffle(initial_indices)
        for i in range(0, num_batches * batch_size, batch_size):
            indices = initial_indices[i: i + batch_size]
            x = [self.corpus[j: j + num_steps] for j in indices]
            y = [self.corpus[j + 1: j + 1 + num_steps] for j in indices]
            yield torch.tensor(x), torch.tensor(y)  # [B, L], [B, L]
    
    # 顺序采样
    def _seq_data_iter_sequential(self):
        batch_size = self.batch_size
        num_steps = self.num_steps
        self.corpus = self.corpus[random.randint(0, num_steps - 1):]
        num_batches = (len(self.corpus) - 1) // batch_size
        Xs = self.corpus[: num_batches * batch_size]
        Ys = self.corpus[1: num_batches * batch_size + 1]
        Xs = torch.tensor(Xs).reshape(batch_size, -1)
        Ys = torch.tensor(Ys).reshape(batch_size, -1)
        num_seqs = Xs.shape[1] // num_steps
        for i in range(0, num_seqs * num_steps, num_steps):
            yield Xs[:, i: num_steps + i], Ys[:, i: i + num_steps]  # [B, L], [B, L]


def load_timemachine(min_freqs=0, token='word'):
    lines = read_file(filename)
    tokens = tokenize(lines, token)
    corpus = [token for line in tokens for token in line]
    vocab = Vocab(corpus, min_freqs)
    return corpus, vocab


class RNNFromScratch:
    def __init__(self, hidden_size, vocab_size, device):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.W_xh, self.W_hh, self.b_h, self.W_hq, self.b_q = self._getParams(device)

    
    def _getParams(self, device):
        W_xh = torch.normal(size=(self.vocab_size, self.hidden_size), mean=0., std=0.01, requires_grad=True, device=device)
        W_hh = torch.normal(size=(self.hidden_size, self.hidden_size), mean=0., std=0.01, requires_grad=True, device=device)
        b_h = torch.zeros(self.hidden_size, requires_grad=True, device=device)
        W_hq = torch.normal(size=(self.hidden_size, self.vocab_size), mean=0., std=0.01, requires_grad=True, device=device)
        b_q = torch.zeros(self.vocab_size, requires_grad=True, device=device)
        return [W_xh, W_hh, b_h, W_hq, b_q]

    
    def forward(self, x, state):  # x:[L, B, V](V:vocab_size), state:[H]
        outputs = []
        H = state
        for t in x:  # t:[B, V]
            H = torch.tanh(torch.mm(t, self.W_xh) + torch.mm(H, self.W_hh) + self.b_h)  # [B, H]
            Y = torch.mm(H, self.W_hq) + self.b_q  # [B, V]
            outputs.append(Y)
        return torch.cat(outputs, dim=0), H  # [B*L, V], [B, H]
    

    def getInitState(self, batch_size, device):
        return torch.zeros((batch_size, self.hidden_size), device=device)
    

def predict(net, prefix, vocab, len_pred, device):  # 先预热得到rnn的隐藏状态
    inputs = torch.tensor(vocab[list(prefix)], device=device)  # [L]
    outputs = [inputs[0]]  # [1]
    inputs = inputs[1:]
    state = net.getInitState(batch_size=1, device=device)
    for i in range(inputs.shape[0]):
        x = F.one_hot(outputs[-1].reshape(1, -1), net.vocab_size).float()  # [1, 1, V]
        x = x.to(device)
        _, state = net.forward(x, state)
        outputs.append(inputs[i])
    for i in range(len_pred):
        x = F.one_hot(outputs[-1].reshape(1, -1), net.vocab_size).float()  # [1, 1, V]
        x = x.to(device)
        y, state = net.forward(x, state)
        outputs.append(y.argmax(dim=1)[0])  # [1, V]->[1]
    return torch.tensor([i.item() for i in outputs])


def sgd(params, lr):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad
            param.grad.zero_()


def grad_clipping(params, theta):
    with torch.no_grad():
        norm = torch.sqrt(torch.sum(torch.tensor([(p.grad ** 2).sum().item() for p in params])))
        if norm > theta:
            for p in params:
                p.grad[:] *= (theta / norm)

    
def train_epoch_squential(net, data_sampler, lr, theta, loss_fn, device):
    total_loss = 0.
    total_num = 0.
    state = net.getInitState(batch_size=1, device=device)
    for x, y in data_sampler.seq_data_sample_fn():
        x = F.one_hot(x.T, net.vocab_size).float().to(device)  # [B, L]->[L, B, V]
        y = y.T.reshape(-1).to(device)  # [L, B]->[L*B]
        state = state.detach_()
        y_hat, state = net.forward(x, state)  # [B*L, H]
        loss = loss_fn(y_hat, y)
        total_loss += loss.detach().clone().item() * y.numel()
        total_num += y.numel()
        loss.backward()
        grad_clipping([net.W_xh, net.W_hh, net.W_hq, net.b_q, net.b_h], theta=theta)
        sgd([net.W_xh, net.W_hh, net.W_hq, net.b_q, net.b_h], lr=lr)
    return math.exp(total_loss / total_num)


def train_epoch_random(net, data_sampler, lr, theta, loss_fn, device):
    total_loss = 0.
    total_num = 0.
    for x, y in data_sampler.seq_data_sample_fn():
        x = F.one_hot(x.T, net.vocab_size).float().to(device)  # [B, L]->[L, B, V]
        y = y.T.reshape(-1).to(device)  # [L, B]->[L*B]
        state = net.getInitState(batch_size=1, device=device)
        y_hat, state = net.forward(x, state)  # [B*L, H]
        loss = loss_fn(y_hat, y)
        total_loss += loss.detach().clone().item() * y.numel()
        total_num += y.numel()
        loss.backward()
        grad_clipping([net.W_xh, net.W_hh, net.W_hq, net.b_q, net.b_h], theta=theta)
        sgd([net.W_xh, net.W_hh, net.W_hq, net.b_q, net.b_h], lr=lr)
    return math.exp(total_loss / total_num)


def train(num_epochs, data_sampler, lr, theta, device, print_every, is_random=False):
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        if not is_random:
            ppl = train_epoch_squential(net, data_sampler, lr, theta, loss_fn, device)
        else:
            ppl = train_epoch_random(net, data_sampler, lr, theta, loss_fn, device)
        if (epoch + 1) % print_every == 0:
            print(f'epoch {epoch + 1}, ppl {ppl:.8f}')



if __name__ == '__main__':
    hidden_size = 512
    lr = 1
    num_epochs = 500
    num_steps = 35
    batch_size = 32
    lines = read_file(filename)
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    corpus = vocab[[token for line in tokens for token in line]]
    corpus = corpus[:10000]
    device = torch.device('cuda:3')
    data_sampler = seq_data_sample(corpus, batch_size, num_steps, True)
    net = RNNFromScratch(hidden_size, len(vocab), device=device)
    train(num_epochs, data_sampler, lr, theta=1., device=device, print_every=1, is_random=True)
    print(''.join(vocab.to_tokens(predict(net, 'time traveller', vocab, 20, device))))