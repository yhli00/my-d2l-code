# 基于注意力机制的自然语言推理：
#     + (1) 首先将句子经过embedding(权重固定，由glove初始化)得到A,B([B, L_A, E], [B, L_B, E])
#     + mlp模块:包含两层，一个隐藏层和一个输出层（输出层的输入维度和输出维度相等）
#     + (2) 将（1）中的结果经过第一个mlp，得到f_A, f_B，基于f_A和f_B计算注意力值，再将注意力值作为权重对A,B加权求和
#        得到beta和alpha([B, L_A, H], [B, L_B, H]),
#     + (3) 将A和beta concate，B和alpha concate，结果通过第二个mlp得到V_A, V_B([B, L_A, H], [B, L_B, H])
#     + (4) 将V_A和V_B在dim=1的维度相加（[B, H]），然后concate([B, 2*H])，然后通过第三个mlp，最后通过一个线性输出层得到结果

# 模型参数：
#     + embed_size = 100, 由glove.6B.100d初始化
#     + 三个mlp的隐藏维度都是200

# 其他超参数：
#     + lr=0.001, num_epochs=4
#     + adam
#     + batch_size=256, 前提和假设的num_steps=50
#     + min_freq=5
#     + dropout=0.2
#     + relu

# 训练集549367个样本，测试集9824个样本



import torch
from torch import nn
from collections import Counter
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F



train_file = '../data/snli_1.0/snli_1.0_train.txt'
test_file = '../data/snli_1.0/snli_1.0_test.txt'
label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
glove_file = '../data/glove/glove.6B.100d.txt'


torch.manual_seed(1234)
device = torch.device('cuda:0')


class PreProcess():
    @staticmethod
    def read_file(is_train=True):
        if is_train:
            filename = train_file
        else:
            filename = test_file
        desc = 'read train data' if is_train else 'read test data'
        with open(filename, 'r') as f:
            lines = f.readlines()[1:]
        lines = [line.strip().lower().split('\t') for line in lines]
        lines = [(line[0].strip(), line[5].strip(), line[6].strip()) for line in lines if line[0].strip() in label_set]
        lines = [(label_set[line[0]], PreProcess.tokenize(PreProcess._add_space(line[1])), 
                 PreProcess.tokenize(PreProcess._add_space(line[2]))) for line in tqdm(lines, desc=desc)]
        labels = [line[0] for line in lines]
        premises = [line[1] for line in lines]
        hypothesises = [line[2] for line in lines]
        return labels, premises, hypothesises
    
    @staticmethod
    def _add_space(text):
        def no_space(letter, pre_letter):
            if letter in set('.?,!') and pre_letter != ' ':
                return True
            return False
        
        text = [' ' + letter if i >= 1 and no_space(letter, text[i - 1]) else letter for i, letter in enumerate(text)]
        return ''.join(text)
    
    @staticmethod
    def tokenize(text):
        return text.split()
    
    @staticmethod
    def truncate_pad(text, num_steps):  # list
        if len(text) > num_steps:
            return text[:num_steps]
        text = text + ['<pad>'] * (num_steps - len(text))
        return text


class Vocab:
    def __init__(self, premises, hypothesises, min_freq=-1, reserved_tokens=[]):
        corpus = [token for line in premises for token in line] + [token for line in hypothesises for token in line]
        counter = Counter(corpus)
        self.unk = 0
        self.idx_to_token = ['<unk>']
        self.token_to_idx = {'<unk>': 0}
        for token in reserved_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1
        for token, freq in counter.items():
            if freq >= min_freq:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, list):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def __len__(self):
        return len(self.idx_to_token)
    
    def to_tokens(self, indices):
        if not isinstance:
            return self.idx_to_token[indices]
        return ' '.join([self.idx_to_token[indic] for indic in indices])
        

class GloveEmbedding:
    def __init__(self, filename):
        self.glove_file = filename
        self.idx_to_token, self.idx_to_vec = self._read_file()
        self.unk = 0
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
    
    def _read_file(self):
        idx_to_token = ['<unk>']
        idx_to_vec = []
        with open(self.glove_file, 'r') as f:
            lines = f.readlines()
        lines = [line.split() for line in lines]
        for line in tqdm(lines, desc='load glove embedding'):
            idx_to_token.append(line[0])
            idx_to_vec.append([float(i) for i in line[1:]])
        idx_to_vec = [[0.] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, torch.tensor(idx_to_vec, dtype=torch.float)
    
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, list):
            return self.idx_to_vec[self.token_to_idx.get(tokens, self.unk)]
        indices = [self.token_to_idx.get(token, self.unk) for token in tokens]
        return self.idx_to_vec[indices]


class nli_dataset(Dataset):
    def __init__(self, num_steps, is_train=True, vocab=None):
        super().__init__()
        self.vocab = None
        if is_train:
            self.labels, self.premises, self.hypothesises = PreProcess.read_file(is_train)
            self.vocab = Vocab(self.premises, self.hypothesises, min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.labels, self.premises, self.hypothesises = PreProcess.read_file(is_train)
        if self.vocab is None:
            self.vocab = vocab
        self.premises = [PreProcess.truncate_pad(line, num_steps) for line in self.premises]
        self.hypothesises = [PreProcess.truncate_pad(line, num_steps) for line in self.hypothesises]
        self.premises = [self.vocab[line] for line in self.premises]
        self.hypothesises = [self.vocab[line] for line in self.hypothesises]
    
    def __getitem__(self, index):
        return (
            torch.tensor(self.labels[index], dtype=torch.long),
            torch.tensor(self.premises[index], dtype=torch.long),
            torch.tensor(self.hypothesises[index], dtype=torch.long)
        )
    
    def __len__(self):
        return len(self.labels)


class MLP(nn.Module):
    def __init__(self, num_inputs, hidden_size, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.mlp(x)


class AttentionModelForNLI(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.mlp_1 = MLP(embed_size, hidden_size, dropout)
        self.mlp_2 = MLP(embed_size * 2, hidden_size, dropout)
        self.mlp_3 = MLP(hidden_size * 2, hidden_size, dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_size, 3)
    
    def forward(self, A, B):  # [B, L_A], [B, L_B]
        A = self.embedding(A)
        B = self.embedding(B)
        f_A = self.mlp_1(A)  # [B, L_A, H]
        f_B = self.mlp_1(B)  # [B, L_B, H]
        A_B_attn = torch.bmm(f_A, f_B.transpose(1, 2))  # [B, L_A, L_B]
        beta = torch.bmm(F.softmax(A_B_attn, dim=-1), B)  # [B, L_A, E]
        alpha = torch.bmm(F.softmax(A_B_attn.transpose(1, 2), dim=-1), A)  # [B, L_B, E]
        V_A = self.mlp_2(torch.cat([A, beta], dim=2))  # [B, L_A, H]
        V_B = self.mlp_2(torch.cat([B, alpha], dim=2))  # [B, L_B, H]
        V_A = torch.sum(V_A, dim=1)
        V_B = torch.sum(V_B, dim=1)
        V = self.mlp_3(torch.cat([V_A, V_B], dim=-1))
        return self.linear(self.dropout(V))


def train_epoch(net, train_data, optimizer, loss_fn, epoch):
    total_loss = 0.
    total_num = 0.
    net.train()
    for label, sentence1, sentence2 in tqdm(train_data, desc=f'train epoch {epoch + 1}'):
        label = label.to(device)
        sentence1 = sentence1.to(device)
        sentence2 = sentence2.to(device)
        y_hat = net(sentence1, sentence2)
        loss = loss_fn(y_hat, label)
        with torch.no_grad():
            total_loss += loss.clone().detach().item() * label.shape[0]
            total_num += label.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / total_num


def test(net, test_data):
    total_num = 0.
    total_correct = 0.
    net.eval()
    for label, sentence1, sentence2 in test_data:
        label = label.to(device)
        sentence1 = sentence1.to(device)
        sentence2 = sentence2.to(device)
        y_hat = net(sentence1, sentence2)
        y_hat = y_hat.argmax(dim=-1)
        with torch.no_grad():
            total_correct += torch.sum(y_hat == label)
            total_num += label.shape[0]
    return total_correct / total_num


def train(net, train_data, test_data, lr, num_epochs, print_every=1):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    for epoch in range(num_epochs):
        loss = train_epoch(net, train_data, optimizer, loss_fn, epoch)
        acc = test(net, test_data)
        if (epoch + 1) % print_every == 0:
            print(f'epoch {epoch + 1}, avg_loss {loss:.4f}, test_acc {acc*100:.2f}%')




if __name__ == '__main__':
    num_epochs = 4
    lr = 0.001
    batch_size = 256
    dropout = 0.2
    embed_size = 100
    hidden_size = 200
    num_steps = 50


    train_set = nli_dataset(num_steps, True)
    vocab = train_set.vocab
    test_set = nli_dataset(num_steps, False, vocab)
    train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_data = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)


    net = AttentionModelForNLI(len(vocab), embed_size, hidden_size, dropout)

    def init_weight(x):
        if type(x) == nn.Linear:
            nn.init.xavier_uniform_(x.weight)

    net.apply(init_weight)
    glove_embedding = GloveEmbedding(glove_file)
    glove_embed = glove_embedding[vocab.idx_to_token]
    net.embedding.weight.data.copy_(glove_embed)
    net.embedding.weight.data.requires_grad = False
    net = net.to(device)


    train(net, train_data, test_data, lr, num_epochs)