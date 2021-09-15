# 使用加性注意力的seq2seq模型实现机器翻译
# 在解码时刻，解码器每个时刻的隐藏状态作为query，编码器的输出作为key和value，将得到的结果和这一时刻的label contvcat起来作为解码器的输入
# 加性注意力：把query和key映射到同一大小（在本例中为hidden_size），经过tanh激活函数，然后经过一次线性变换使维度变为1
# embed_size=32
# hidden_size=32
# 编码器解码器使用两层单项gru
# dropout=0.1
# lr=0.05
# 使用adam优化器
# batch_size=64
# num_steps=10
# num_epochs=500
# 数据集共有190206条训练数据



import math
import torch
from collections import Counter
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch import optim


torch.manual_seed(123456)
file_path = '../data/fra.txt'
device = torch.device('cuda:0')


class PreProcess():
    def __init__(self):
        pass

    def read_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [line[: line.index('CC-BY')].strip().lower() for line in lines]
        lines = [line.replace('\u202f', ' ').replace('\xa0', ' ').strip() for line in lines]
        lines = [self._addSpace(line) for line in lines]
        return lines
        
    def _addSpace(self, line):
        def no_space(letter, pre_letter):
            if letter in set(',.!?') and pre_letter != ' ':
                return True
            return False
        letters = [' ' + j if i > 0 and no_space(j, line[i - 1]) else j for i, j in enumerate(line)]
        return ''.join(letters)
    
    def tokenize(self, lines, num_examples=None):
        cnt = 0
        source = []
        target = []
        for line in lines:
            cnt += 1
            part1, part2 = line.split('\t')
            source.append(part1.split())
            target.append(part2.split())
            if num_examples is not None and cnt > num_examples:
                break
        return source, target

    def truncate_pad(self, line, num_steps):
        line = line + ['<eos>']
        if len(line) > num_steps:
            return line[: num_steps]
        else:
            line = line + ['<pad>' for _ in range(num_steps - len(line))]
        return line

class Vocab:
    def __init__(self, tokens, min_freqs=-1, reserved_tokens=[]):
        self.corpus = [token for line in tokens for token in line]
        counter = Counter(self.corpus)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.unk = 0
        self.idx_to_token = ['<unk>']
        self.token_to_idx = {'<unk>': 0}
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


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.embeding = nn.Linear(vocab_size, embed_size)
        self.gru = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

    def forward(self, x):  # x: [L, B, V] state默认是0
        x = self.embeding(x)
        output, state = self.gru(x)  # [L, B, H], [num_layers, B, H]
        return output, state


def seq_mask(x, valid_len, value=0):  # [B, L], [B]
    pos = torch.arange(x.shape[1], device=x.device).repeat(x.shape[0], 1)
    valid_len = valid_len.reshape(-1, 1).repeat(1, x.shape[1])
    mask = pos < valid_len
    x[~mask] = value
    return x


def masked_softmax(q_k_attn, valid_len, value=-1e9):  # [B, q, k], [B]
    shape = q_k_attn.shape
    valid_len = valid_len.repeat_interleave(q_k_attn.shape[1])
    masked_attn = seq_mask(q_k_attn.reshape(-1, shape[-1]), valid_len, value)
    return F.softmax(masked_attn.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    def __init__(self, q_size, k_size, hidden_size, dropout):
        super().__init__()
        self.W_q = nn.Linear(q_size, hidden_size, bias=False)
        self.W_k = nn.Linear(k_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, q, k, v, valid_len):
        q = self.W_q(q)  # [B, num_q, H]
        k = self.W_k(k)  # [B, num_k, H]
        q_k = q.unsqueeze(2) + k.unsqueeze(1)  # [B, num_q, num_k, H]
        q_k = self.W_v(q_k).squeeze(-1)  # [B, num_q, num_k]
        q_k_attn = self.dropout(masked_softmax(q_k, valid_len))
        return torch.bmm(q_k_attn, v)  # [B, num_q, V]



class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.embeding = nn.Linear(vocab_size, embed_size)
        self.gru = nn.GRU(input_size=embed_size + hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.attention = AdditiveAttention(hidden_size, hidden_size, hidden_size, dropout=dropout)
    
    def get_init_state(self, enc_outputs, enc_valid_len):
        enc_output, hidden_state = enc_outputs
        return enc_output, hidden_state, enc_valid_len
    
    def forward(self, x, state):  # x:[L, B, V]
        output = []
        x = self.embeding(x)  # [L, B, E]
        enc_output, hidden_state, enc_valid_len = state
        for t in x:  # [B, E]
            query = hidden_state[-1].unsqueeze(0).transpose(0, 1)  # [B, 1, H]
            context = self.attention(query, enc_output.transpose(0, 1), enc_output.transpose(0, 1), enc_valid_len)  # [B, 1, H]
            t = torch.cat([t.unsqueeze(0), context.transpose(0, 1)], dim=-1)
            dec_output, hidden_state = self.gru(t, hidden_state)
            output.append(dec_output)
        output = torch.cat(output, dim=0).transpose(0, 1)  # [B, L, H]
        return self.out(output), hidden_state

class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, source_vocab_size, target_vocab_size, dropout):
        super().__init__()
        self.encoder = Encoder(source_vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(target_vocab_size, embed_size, hidden_size, num_layers, dropout)
    
    def forward(self, x, y, enc_valid_len):
        (enc_output, state) = self.encoder(x)
        state = self.decoder.get_init_state((enc_output, state), enc_valid_len)
        return self.decoder(y, state)  # [B, L, V]


def train_epoch(net, data_iter, source_vocab, target_vocab, optimizer, loss_fn, max_norm):
    net.train()
    total_loss = 0.
    total_words = 0.
    for x, y in data_iter:
        x = x.to(device)
        y = y.to(device)
        x_valid_len = torch.sum(x != source_vocab['<pad>'], dim=1)
        y_valid_len = torch.sum(y != target_vocab['<pad>'], dim=1)
        bos = torch.tensor([target_vocab['<bos>']], device=device).reshape(-1, 1).repeat(y.shape[0], 1)  # [B, 1]
        y_input = torch.cat([bos, y[:, :-1]], dim=-1)
        x = F.one_hot(x.T, len(source_vocab)).float()  # [L, B, V]
        y_input = F.one_hot(y_input.T, len(target_vocab)).float()  # [L, B, V]
        y_hat, _ = net(x, y_input, x_valid_len)
        loss = loss_fn(y_hat, y, y_valid_len)
        with torch.no_grad():
            total_loss += loss.detach().clone().sum().item()
            total_words += torch.sum(y.detach().clone() != target_vocab['<pad>']).item()
        optimizer.zero_grad()
        loss.sum().backward()
        clip_grad_norm_(net.parameters(), max_norm)
        optimizer.step()
    return total_loss / total_words


class MaskedCELoss(nn.CrossEntropyLoss):
    def forward(self, y_hat, y, valid_len):  # [B, L, V], [B, L], [B]
        self.reduction = 'none'
        loss = super(MaskedCELoss, self).forward(y_hat.transpose(1, 2), y)  # [B, L]
        mask = torch.ones(loss.shape, device=loss.device, dtype=loss.dtype)
        mask = seq_mask(mask, valid_len, value=0.)
        return mask * loss



def train(net, lr, num_epochs, data_iter, source_vocab, target_vocab, max_norm, print_every=1):
    loss_fn = MaskedCELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        loss = train_epoch(net, data_iter, source_vocab, target_vocab, optimizer, loss_fn, max_norm)
        if (epoch + 1) % print_every == 0:
            print(f'epoch {epoch + 1}, loss_per_word {loss:.6f}')


def predict(src, source_vocab, target_vocab, net, num_steps, pre_pro):
    net.eval()
    output = []
    src = src.strip().lower().split()
    enc_valid_len = torch.tensor([len(src)], device=device)
    src = pre_pro.truncate_pad(src, num_steps)
    src_tokens = source_vocab[src]
    src_tokens = torch.tensor([src_tokens])  # [1, L]
    enc_input = F.one_hot(src_tokens.T, len(source_vocab)).float().to(device)  # [L, 1, V]
    enc_output, hidden_state = net.encoder(enc_input)
    dec_input = torch.tensor([[target_vocab['<bos>']]])  # [1, 1]
    for _ in range(num_steps):
        dec_input = F.one_hot(dec_input, len(target_vocab)).float().to(device)
        dec_input, hidden_state = net.decoder(dec_input, (enc_output, hidden_state, enc_valid_len))
        dec_input = dec_input.argmax(dim=-1)  # [1, 1]
        if dec_input.item() == target_vocab['<eos>']:
            break
        output.append(dec_input.item())
    return target_vocab.to_tokens(output)


def bleu(pred, label, k):
    pred = pred.strip().split()
    label = label.strip().split()
    score = math.exp(min(0, 1 - len(label) / len(pred)))
    for n in range(1, k + 1):
        ngram_cnt = {}
        match = 0
        for i in range(0, len(pred) - n + 1):
            ngram = ''.join(pred[i: i + n])
            ngram_cnt[ngram] = ngram_cnt.get(ngram, 0) + 1
        for i in range(0, len(label) - n + 1):
            ngram = ''.join(label[i: i + n])
            if ngram in ngram_cnt and ngram_cnt[ngram] > 0:
                ngram_cnt[ngram] -= 1
                match += 1
        score *= math.pow(match / (len(pred) - n + 1), math.pow(1 / 2, n))
    return score



if __name__ == '__main__':
    num_examples = 10000
    batch_size = 1024 * 2
    num_steps = 10
    lr = 0.005
    max_norm = 1.0
    embed_size = 32
    hidden_size = 32
    num_layers = 2
    dropout = 0.1
    num_epochs = 500


    pre_pro = PreProcess()
    lines = pre_pro.read_file(file_path)
    source, target = pre_pro.tokenize(lines, num_examples=num_examples)
    source_vocab = Vocab(source, min_freqs=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    target_vocab = Vocab(target, min_freqs=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    source = [pre_pro.truncate_pad(line, num_steps=num_steps) for line in source]
    target = [pre_pro.truncate_pad(line, num_steps=num_steps) for line in target]
    source = [source_vocab[line] for line in source]
    target = [target_vocab[line] for line in target]


    train_data = DataLoader(TranslationDataset(source, target), batch_size=batch_size, shuffle=True, num_workers=4)


    net = EncoderDecoder(embed_size, hidden_size, num_layers, len(source_vocab), len(target_vocab), dropout=0.1)
    net = net.to(device)

    train(net, lr, num_epochs, train_data, source_vocab, target_vocab, max_norm, print_every=1)
    

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .', 'run !']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .', 'prenez vos jambes à vos cous !']
    for i in range(len(engs)):
        pred = predict(engs[i], source_vocab, target_vocab, net, num_steps, pre_pro)
        print(engs[i], '-->', pred, '--> bleu2: ', bleu(pred, fras[i], k=2))

