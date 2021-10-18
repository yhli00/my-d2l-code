# 基于BiLSTM的情感分析：
#     + 使用2层双向lstm，将bilstm第一个时间步和最后一个时间步的输出concate起来，送到线性输出层
#     + 使用glove.6B.100d词向量
#     + 训练集和测试集的大小都是25000
#     + adam, lr=0.01, CrossEntropyLoss
#     + 使用xvaier初始化
#     + num_step = 500
#     + batch_size = 64
#     + 词嵌入和bilstm的隐层维度都是100
#     + truncate and pad
#     + 去除出现次数小于5次的单词


import os
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter


train_data_dir = '../data/aclImdb/train'
test_data_dir = '../data/aclImdb/test'
glove_data_path = '../data/glove/glove.6B.100d.txt'


torch.manual_seed(1234)
device = torch.device('cuda:0')


class PreProcess:
    @staticmethod
    def read_file(data_dir):
        labels = []
        texts = []
        if 'train' in data_dir:
            desc = 'read train data'
        else:
            desc = 'read test data'
        for label in ['pos', 'neg']:
            filenames = os.listdir(os.path.join(data_dir, label))
            for filename in tqdm(filenames, desc=desc + ' ' + label):
                with open(os.path.join(data_dir, label, filename), 'r') as f:
                    text = f.read()
                    text = text.lower().strip()
                    text = PreProcess._add_space(text)
                    text = PreProcess._tokenize(text)
                if label == 'pos':
                    labels.append(1)
                    texts.append(text)
                else:
                    labels.append(0)
                    texts.append(text)
        return labels, texts
    
    @staticmethod
    def _add_space(text):
        def _is_without_space(letter, pre_letter):
            if letter in set('.,!?"') and pre_letter != ' ':
                return True
            if letter != ' ' and pre_letter in set('.,!?"'):
                return True
            return False
        
        add_space_text = [' ' + j if i >= 1 and _is_without_space(j, text[i - 1]) else j for i, j in enumerate(text)]
        return ''.join(add_space_text)
    
    @staticmethod
    def _tokenize(text):
        return text.split()
    
    @staticmethod
    def truncate_and_pad(text_list, max_len):
        if len(text_list) >= max_len:
            return text_list[:max_len]
        text_list = text_list + ['<pad>'] * (max_len - len(text_list))
        return text_list



class Vocab:
    def __init__(self, texts, min_freqs=-1, reserved_tokens=[]):
        corpus = [token for text in texts for token in text]
        counter = Counter(corpus)
        self.unk = 0
        self.idx_to_token = ['<unk>']
        self.token_to_idx = {'<unk>': 0}
        for token in reserved_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1
        for token, freq in counter.items():
            if freq >= min_freqs:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, list):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def __len__(self):
        return len(self.idx_to_token)

    def to_tokens(self, indices):
        if not isinstance(indices, list):
            return self.idx_to_token[indices]
        return [self.idx_to_token[indic] for indic in indices]


class GloveEmbedding:
    def __init__(self):
        self.idx_to_token, self.idx_to_vec = self._read_file()
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        self.idx_to_vec = torch.tensor(self.idx_to_vec, dtype=torch.float)
        self.unk = 0

    def _read_file(self):
        idx_to_vec = []
        idx_to_token = ['<unk>']
        with open(glove_data_path, 'r') as f:
            lines = f.readlines()
        for line in tqdm(lines, desc='load glove embeddings'):
            line = line.strip().split()
            idx_to_token.append(line[0])
            vec = [float(token) for token in line[1:]]
            idx_to_vec.append(vec)
        idx_to_vec = [[0.] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, idx_to_vec
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, list):
            return self.idx_to_vec[self.token_to_idx.get(tokens, self.unk)]
        indices = [self.token_to_idx.get(token, self.unk) for token in tokens]
        return self.idx_to_vec[indices]
    
    def __len__(self):
        return len(self.idx_to_token)


class imdb_dataset(Dataset):
    def __init__(self, max_len, is_train=True, vocab=None):
        super().__init__()
        self.vocab = None
        if is_train:
            labels, texts = PreProcess.read_file(train_data_dir)
            self.vocab = Vocab(texts, min_freqs=5, reserved_tokens=['<pad>'])
        else:
            labels, texts = PreProcess.read_file(test_data_dir)
        if self.vocab is None:
            self.vocab = vocab
        texts = [PreProcess.truncate_and_pad(text, max_len) for text in texts]
        self.texts = [self.vocab[text] for text in texts]
        self.labels = labels
    
    def __getitem__(self, index):
        return torch.tensor(self.texts[index], dtype=torch.long), torch.tensor(self.labels[index], dtype=torch.long)  # [B, L], [B]
    
    def __len__(self):
        return len(self.labels)


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.bilstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
        self.output = nn.Linear(4 * hidden_size, 2)
    
    def forward(self, x):  # [L, B]
        x_embed = self.embedding(x)
        output, _ = self.bilstm(x_embed)  # [L, B, 2*H]
        output = torch.cat([output[0], output[-1]], dim=1)  # [B, 4*H]
        return self.output(output)  # [B, 2]
    

def train_epoch(train_data, net, optimizer, epoch, loss_fn):
    net.train()
    total_loss = 0.
    total_num = 0.
    for x, y in tqdm(train_data, desc=f'train epoch {epoch + 1}'):
        x = x.to(device)
        y = y.to(device)
        y_hat = net(x.T)
        loss = loss_fn(y_hat, y)
        with torch.no_grad():
            total_loss += loss.detach().clone().item() * y.shape[0]
            total_num += y.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / total_num


def test(test_data, net, epoch):
    net.eval()
    with torch.no_grad():
        total_num = 0.
        total_correct = 0.
        for x, y in tqdm(test_data, desc=f'test epoch {epoch + 1}'):
            x = x.to(device)
            y = y.to(device)
            y_hat = net(x.T)
            y_hat = y_hat.argmax(dim=-1)
            total_correct += torch.sum(y.type(y_hat.dtype) == y_hat)
            total_num += y.shape[0]
    return total_correct / total_num


def train(train_data, test_data, lr, net, num_epochs, print_every=1):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    for epoch in range(num_epochs):
        avg_loss = train_epoch(train_data, net, optimizer, epoch, loss_fn)
        if (epoch + 1) % print_every == 0:
            test_acc = test(test_data, net, epoch)
            print(f'epoch {epoch + 1}, avg_loss {avg_loss:.5f}, test_acc {test_acc*100:.2f}%')
        

def predict(net, vocab, pred):
    pred = pred.lower().strip().split()
    pred = vocab[pred]
    pred = torch.tensor(pred, dtype=torch.long, device=device)
    pred = pred.unsqueeze(0)
    y_hat = net(pred.T)
    y_hat = y_hat.argmax(dim=1)
    if y_hat.item() == 1:
        return 'pos'
    return 'neg'



if __name__ == '__main__':
    batch_size = 256
    max_len = 500
    min_freqs = 5
    embed_size = 100
    hidden_size = 100
    num_layers = 2
    num_epochs = 5
    lr = 0.01
    

    train_set = imdb_dataset(max_len, True)
    vocab = train_set.vocab
    test_set = imdb_dataset(max_len, False, vocab)
    train_data = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=4)
    test_data = DataLoader(test_set, shuffle=True, batch_size=batch_size, num_workers=4)
    

    net = BiLSTM(len(vocab), embed_size, hidden_size, num_layers)
    
    def init_weight(x):
        if type(x) == nn.Linear:
            nn.init.xavier_uniform_(x.weight)
        if type(x) == nn.LSTM:
            for param in x._flat_weights_names:
                if 'weight' in param:
                    nn.init.xavier_uniform_(x._parameters[param])

    net.apply(init_weight)
    glove_embedding = GloveEmbedding()
    embed = glove_embedding[vocab.idx_to_token]
    net.embedding.weight.data.copy_(embed)
    net.embedding.weight.requires_grad = False
    net = net.to(device)


    train(train_data, test_data, lr, net, num_epochs)

    for i in ['this movie is so great', 'this movie is so bad']:
        print(i, ' : ', predict(net, vocab, i))

    