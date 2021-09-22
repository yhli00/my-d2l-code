# transformer的pytorch实现
# 注意要点：
#     + dropout的使用位置：
#         - 计算位置编码之后
#         - 缩放点积注意力中计算q_k_atten之后
#         - AddNorm之前
#     + 基于位置的前馈网络使用relu激活函数
#     + 计算多头注意力时nn.Linear的bias为False
#     + 加上位置编码之前需要吧词嵌入乘以词嵌入维度的平方根
#     + 为了加速计算，多头注意力中每个头的维度是hidden_size / num_heads
# 参数：
#     + hidden_size = 32
#     + num_layers = 2
#     + dropout = 0.1
#     + batch_size = 64
#     + num_steps = 10
#     + lr = 0.005
#     + num_epochs = 200
#     + ffn_input_size, ffn_hidden_size = 32, 64
#     + num_heads = 4
#     + q_size, k_size, v_size = 32, 32, 32
#     + adam优化器


import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
from collections import Counter
import torch.nn.functional as F
import math


torch.manual_seed(888888)
device = torch.device('cuda:1')
file_path = '../data/fra.txt'



class PreProcess():
    def __init__(self):
        pass
    
    def read_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [line[: line.index('CC-BY')] for line in lines]
        lines = [line.strip().lower() for line in lines]
        lines = [line.replace('\u202f', ' ').replace('\xa0', ' ').strip() for line in lines]
        lines = [self._addSpace(line) for line in lines]
        return lines

    def _addSpace(self, line):
        def no_space(letter, pre_letter):
            if letter in set('.,?!') and pre_letter != ' ':
                return True
            return False
        line = [' ' + j if i > 0 and no_space(j, line[i - 1]) else j for i, j in enumerate(line)]
        return ''.join(line)
    
    def tokenize(self, lines, num_examples):
        cnt = 0
        source = []
        target = []
        for line in lines:
            part1, part2 = line.split('\t')
            source.append(part1.strip().split())
            target.append(part2.strip().split())
            cnt += 1
            if cnt >= num_examples:
                break
        return source, target
    
    def truncate_pad(self, line, num_steps):
        line = line + ['<eos>']
        if len(line) >= num_steps:
            line = line[: num_steps]
            return line
        else:
            line = line + ['<pad>' for _ in range(num_steps - len(line))]
            return line


class Vocab:
    def __init__(self, lines, min_freqs=2, reserved_tokens=[]):
        self.corpus = [token for line in lines for token in line]
        counter = Counter(self.corpus)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.unk = 0
        self.token_to_idx = {'<unk>': 0}
        self.idx_to_token = ['<unk>']
        for t in reserved_tokens:
            self.idx_to_token.append(t)
            self.token_to_idx[t] = len(self.idx_to_token) - 1
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
        return ' '.join([self.idx_to_token[indic] for indic in indices])


class TranslationDataset(Dataset):
    def __init__(self, source, target):
        super().__init__()
        self.source = source
        self.target = target
    
    def __len__(self):
        return len(self.source)
    
    def __getitem__(self, index):
        return torch.tensor(self.source[index]), torch.tensor(self.target[index])


def seq_mask(x, valid_len, value):  # [B, L], [B]
    position = torch.arange(x.shape[1], device=device).repeat(x.shape[0], 1)
    mask = position < valid_len.reshape(-1, 1).repeat(1, x.shape[1])
    x[~mask] = value
    return x

def masked_softmax(x, valid_len, value=-1e9):  # [B, num_q, num_k], [B, num_q]或None
    if valid_len is None:
        return F.softmax(x, dim=-1)
    shape = x.shape
    valid_len_shape = valid_len.shape
    x = x.reshape(-1, shape[-1])
    x_valid_len = valid_len.reshape(-1)
    if x.shape[0] != x_valid_len.shape[0]:
        print(shape)
        print(valid_len_shape)
    x = seq_mask(x, x_valid_len, value)
    x = x.reshape(shape)  # [B, Q, K]
    return F.softmax(x, dim=-1)


class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, q, k, v, valid_len):  # [B, num_q, Q], [B, num_k, Q], [B, num_k, V], [B, num_q]
        q_k_attn = torch.bmm(q, k.transpose(1, 2))
        q_k_attn = masked_softmax(q_k_attn, valid_len)
        return torch.bmm(self.dropout(q_k_attn), v)

class MultiHeadAttention(nn.Module):
    def __init__(self, q_size, k_size, v_size, hidden_size, num_heads, dropout):
        super().__init__()
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(q_size, hidden_size, bias=False)
        self.W_k = nn.Linear(k_size, hidden_size, bias=False)
        self.W_v = nn.Linear(v_size, hidden_size, bias=False)
        self.W_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.num_heads = num_heads
    
    def forward(self, q, k, v, valid_len):
        q = self.W_q(q)  # [B, num_q, H]
        k = self.W_k(k)
        v = self.W_v(v)
        q = self._transpose_qkv(q)  # [B * heads, num_q, H / heads]
        k = self._transpose_qkv(k)
        v = self._transpose_qkv(v)
        if valid_len is not None:  # 训练模式
            k_valid_len = valid_len.repeat_interleave(self.num_heads, dim=0)
        else:  # 预测模式
            k_valid_len = None
        output = self.attention(q, k, v, k_valid_len)  # [B * heads, num_q, H / heads]
        output = self._transpose_output(output)
        return self.W_o(output)  # [B, num_q, H]
        

    def _transpose_qkv(self, q):
        shape = q.shape
        q = q.reshape(shape[0], shape[1], self.num_heads, -1)
        q = q.transpose(1, 2)
        return q.reshape(-1, q.shape[2], q.shape[3])
    

    def _transpose_output(self, output):  # [B * heads, num_q, H / heads]
        shape = output.shape
        output = output.reshape(-1, self.num_heads, shape[1], shape[2])  # [B, heads, num_q, H / heads]
        output = output.transpose(1, 2)
        return output.reshape(output.shape[0], output.shape[1], -1)  # [B, num_q, H]


class PositionalEncoding(nn.Module):
    def __init__(self, dropout, hidden_size, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.P = torch.zeros((1, max_len, hidden_size // 2), device=device)
        self.P[0] = torch.arange(max_len, device=device).reshape(-1, 1) / torch.pow(
            10000,
            torch.arange(0, hidden_size, 2, device=device) / hidden_size
        )

    def forward(self, x):  # [B, L, H]
        x[:, :, ::2] = x[:, :, ::2] + torch.sin(self.P[:, :x.shape[1], :])
        x[:, :, 1::2] = x[:, :, 1::2] + torch.cos(self.P[:, :x.shape[1], :])
        return self.dropout(x)


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_input_size, ffn_hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(ffn_input_size, ffn_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ffn_hidden_size, ffn_input_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class AddNorm(nn.Module):
    def __init__(self, dropout, norm_shape):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(norm_shape)
    
    def forward(self, x, y):
        return self.layer_norm(self.dropout(y) + x)


class EncoderBlock(nn.Module):
    def __init__(self, q_size, k_size, v_size, hidden_size, num_heads, dropout, ffn_input_size, ffn_hidden_size):
        super().__init__()
        self.attention = MultiHeadAttention(q_size, k_size, v_size, hidden_size, num_heads, dropout)
        self.layer_norm1 = AddNorm(dropout, hidden_size)
        self.ffn = PositionWiseFFN(ffn_input_size, ffn_hidden_size)
        self.layer_norm2 = AddNorm(dropout, hidden_size)
    
    def forward(self, enc_output, valid_len):  # [B, L, H], [B, L]
        attention = self.attention(enc_output, enc_output, enc_output, valid_len)
        attention = self.layer_norm1(enc_output, attention)
        ffn = self.ffn(attention)
        return self.layer_norm2(attention, ffn)


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_heads, dropout, ffn_input_size, ffn_hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_encoding = PositionalEncoding(dropout, hidden_size)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                str(i),
                EncoderBlock(hidden_size, hidden_size, hidden_size, hidden_size, num_heads, dropout, 
                                      ffn_input_size, ffn_hidden_size)
            )
        
    def forward(self, x, valid_len):  # [B, L, V], [B, L]
        x = self.embedding(x)
        x = x * math.sqrt(x.shape[-1])
        x = self.position_encoding(x)
        for blk in self.blks:
            x = blk(x, valid_len)
        return x
    

class DecoderBlock(nn.Module):
    def __init__(self, q_size, k_size, v_size, hidden_size, num_heads, dropout, ffn_input_size, ffn_hidden_size):
        super().__init__()
        self.attention1 = MultiHeadAttention(q_size, k_size, v_size, hidden_size, num_heads, dropout)
        self.layer_norm1 = AddNorm(dropout, hidden_size)
        self.attention2 = MultiHeadAttention(q_size, k_size, v_size, hidden_size, num_heads, dropout)
        self.layer_norm2 = AddNorm(dropout, hidden_size)
        self.ffn = PositionWiseFFN(ffn_input_size, ffn_hidden_size)
        self.layer_norm3 = AddNorm(dropout, hidden_size)
        
    
    def forward(self, dec_output, enc_output, enc_valid_len):
        if self.training:  # 训练
            dec_valid_len = torch.arange(1, dec_output.shape[1] + 1, device=device).repeat(dec_output.shape[0], 1)
        else:  # 预测
            dec_valid_len = None
        attention1 = self.attention1(dec_output, dec_output, dec_output, dec_valid_len)
        attention1 = self.layer_norm1(dec_output, attention1)
        attention2 = self.attention2(attention1, enc_output, enc_output, enc_valid_len)
        attention2 = self.layer_norm2(attention1, attention2)
        y = self.ffn(attention2)
        return self.layer_norm3(attention2, y)


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_heads, dropout, num_layers, ffn_input_size, ffn_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_encoding = PositionalEncoding(dropout, hidden_size)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                str(i),
                DecoderBlock(hidden_size, hidden_size, hidden_size, hidden_size, num_heads, dropout, ffn_input_size, ffn_hidden_size)
            )
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, enc_output, enc_valid_len):
        x = self.embedding(x)
        x = x * math.sqrt(x.shape[-1])
        x = self.position_encoding(x)
        for blk in self.blks:
            x = blk(x, enc_output, enc_valid_len)
        return self.output_layer(x)


class TransformerEncoderDecoder(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, hidden_size, num_heads, dropout, num_layers, ffn_input_size, ffn_hidden_size):
        super().__init__()
        self.encoder = TransformerEncoder(source_vocab_size, hidden_size, num_heads, dropout, ffn_input_size, ffn_hidden_size, num_layers)
        self.decoder = TransformerDecoder(target_vocab_size, hidden_size, num_heads, dropout, num_layers, ffn_input_size, ffn_hidden_size)

    def forward(self, x, y, enc_valid_len):  # [B, L], [B, L]
        enc_output = self.encoder(x, enc_valid_len)  # [B, L, H]
        return self.decoder(y, enc_output, enc_valid_len)  # [B, L, V]


class MaskedCELoss(nn.CrossEntropyLoss):
    def forward(self, y_hat, y, y_valid_len):
        self.reduction = 'none'
        loss = super(MaskedCELoss, self).forward(y_hat.transpose(1, 2), y)
        mask = torch.ones(y.shape, device=device)
        mask = seq_mask(mask, y_valid_len, value=0)
        return mask * loss


def train_epoch(net, optimizer, loss_fn, data_iter, source_vocab, target_vocab, max_norm):
    net.train()
    total_loss = 0.
    total_words = 0.
    for x, y in data_iter:
        x = x.to(device)
        y = y.to(device)
        bos = torch.tensor([target_vocab['<bos>']] * y.shape[0], device=device).reshape(-1, 1)
        y_input = torch.cat([bos, y[:, :-1]], dim=-1)
        x_valid_len = torch.sum(x != source_vocab['<pad>'], dim=-1).reshape(-1, 1)
        x_valid_len = x_valid_len.repeat(1, x.shape[-1])  # [B, L]
        y_valid_len = torch.sum(y != target_vocab['<pad>'], dim=-1)  # [B]
        y_hat = net(x, y_input, x_valid_len)
        loss = loss_fn(y_hat, y, y_valid_len)
        with torch.no_grad():
            total_loss += loss.detach().clone().sum().item()
            total_words += y_valid_len.sum().item()
        optimizer.zero_grad()
        loss.sum().backward()
        clip_grad_norm_(net.parameters(), max_norm=max_norm)
        optimizer.step()
    return total_loss / total_words


def train(net, lr, num_epochs, data_iter, source_vocab, target_vocab, max_norm, print_every=1):
    loss_fn = MaskedCELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        loss_per_word = train_epoch(net, optimizer, loss_fn, data_iter, source_vocab, target_vocab, max_norm)
        if (epoch + 1) % print_every == 0:
            print(f'epoch {epoch + 1}, loss_per_word {loss_per_word:.5f}')


def predict(src, net, source_vocab, target_vocab, num_steps):
    net.eval()
    src = src.strip().lower()
    token = src.split() + ['<eos>']
    enc_len = torch.tensor([[len(token)]], device=device)  # [B, 1]
    enc_valid_len = enc_len.repeat(1, len(token))  # [B, L]
    enc_input = torch.tensor([source_vocab[token]], device=device)  # [B, num_q]
    enc_output = net.encoder(enc_input, enc_valid_len)
    dec_input = torch.tensor([[target_vocab['<bos>']]], device=device)
    for _ in range(num_steps):
        enc_valid_len = enc_len.repeat(1, dec_input.shape[1])  # [B, num_q]
        dec_output = net.decoder(dec_input, enc_output, enc_valid_len)  # [B, L, H]
        pred = dec_output.argmax(dim=-1)[:, -1:]  # [B, 1]
        if pred.item() == target_vocab['<eos>']:
            break
        dec_input = torch.cat([dec_input, pred], dim=-1)
    dec_input = dec_input[:, 1:]
    return target_vocab.to_tokens(dec_input.squeeze(0).tolist())


def bleu(pred, label, k):
    pred = pred.strip().lower().split()
    label = label.strip().lower().split()
    score = math.exp(min(0, 1 - len(label) / len(pred)))
    for n in range(1, k + 1):
        ngram_dict = {}
        match = 0
        for i in range(0, len(pred) - n + 1):
            ngram_dict[''.join(pred[i: i + n])] = ngram_dict.get(''.join(pred[i: i + n]), 0) + 1
        for i in range(0, len(label) - n + 1):
            if ''.join(label[i: i + n]) in ngram_dict and ngram_dict[''.join(label[i: i + n])] > 0:
                ngram_dict[''.join(label[i: i + n])] -= 1
                match += 1
        score *= math.pow(match / (len(pred) - n + 1), math.pow(1 / 2, n))
    return score
    


if __name__ == '__main__':
    num_examples = 10000
    batch_size = 256 * 16
    num_steps = 10
    hidden_size = 32
    dropout = 0.1
    num_layers = 2
    num_heads = 4
    ffn_input_size = 32
    ffn_hidden_size = 64
    lr = 0.005
    max_norm = 0.1
    num_epochs = 1000


    pre_pro = PreProcess()
    lines = pre_pro.read_file(file_path)
    source, target = pre_pro.tokenize(lines, num_examples)
    source_vocab = Vocab(source, min_freqs=2, reserved_tokens=['<pad>', '<eos>', '<bos>'])
    target_vocab = Vocab(target, min_freqs=2, reserved_tokens=['<pad>', '<eos>', '<bos>'])
    print(len(source_vocab))
    print(len(target_vocab))
    source = [pre_pro.truncate_pad(line, num_steps) for line in source]
    target = [pre_pro.truncate_pad(line, num_steps) for line in target]
    source = [source_vocab[line] for line in source]
    target = [target_vocab[line] for line in target]
    
    
    train_data = DataLoader(TranslationDataset(source, target), batch_size=batch_size, shuffle=True, num_workers=4)


    net = TransformerEncoderDecoder(len(source_vocab), len(target_vocab), hidden_size, num_heads, dropout,
                                    num_layers, ffn_input_size, ffn_hidden_size)
    net = net.to(device)


    train(net, lr, num_epochs, train_data, source_vocab, target_vocab, max_norm, print_every=1)


    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .', 'run !']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .', 'prenez vos jambes à vos cous !']
    for i in range(len(engs)):
        pred = predict(engs[i], net, source_vocab, target_vocab, num_steps)
        print(engs[i], '-->', pred, '--> bleu2: ', bleu(pred, fras[i], k=2))