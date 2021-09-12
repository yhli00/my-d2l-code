# 英语到法语的机器翻译模型
# 训练数据取前600条英-法翻译对
# 编码器、解码器都使用单向、双层GRU，hidden_size=32, embed_size=32，GRU dropout=0.1
# 解码器最后一个时刻的隐藏层状态作为上下文向量，上下文向量作为解码器的初始隐藏状态，同时与label数据concat起来作为解码器的输入

# 使用adam优化器，lr=0.005
# num_steps=10
# num_epochs=300
# 使用bleu2测试
# max_norm=1.0



import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch import optim
import torch.nn.functional as F
import math



torch.manual_seed(123456)
device = torch.device('cuda:0')
file_path = '../data/fra.txt'


class rowDataPreProcess():
    def __init__(self):
        pass
    
    def readFile(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [line[: line.index('CC-BY')] for line in lines]
        return [line.replace('\u202f', ' ').replace('\xa0', ' ').lower().strip() for line in lines]  # 去除不间断空格
    
    def addSpace(self, line):  # 在字母和标点之间添加空格
        def no_space(letter, pre_letter):
            if letter in set('.,?!') and pre_letter != ' ':
                return True
            return False
        
        line = [' ' + c if i > 0 and no_space(c, line[i - 1]) else c for i, c in enumerate(line)]
        return ''.join(line)
    
    def tokenize(self, lines, num_examples=None):  # num_examples:分词多少个句子
        cnt = 0
        source = []
        target = []
        for line in lines:
            cnt += 1
            if num_examples is not None and cnt > num_examples:
                break
            part1, part2 = line.split('\t')
            source.append(part1.split())
            target.append(part2.split())
        return source, target
    
    def truncate_pad(self, line, num_steps):  # line是分词后的列表
        if len(line) > num_steps:
            return line[: num_steps]
        line += ['<pad>' for _ in range(num_steps - len(line))]
        return line
        

class Vocab:
    def __init__(self, lines, min_freqs=-1, reserved_tokens=None):   # lines是分词后的列表
        self.corpus = [token for line in lines for token in line]
        counter = Counter(self.corpus)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.unk = 0
        self.idx_to_token = ['<unk>']
        self.token_to_idx = {'<unk>': 0}
        for token in reserved_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1
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
        

def build_data(pre_pro, source, target, source_vocab, target_vocab, num_steps):  # 将字符列表编程数字列表
    source = [line + ['<eos>'] for line in source]
    target = [line + ['<eos>'] for line in target]
    source = [pre_pro.truncate_pad(line, num_steps) for line in source]
    target = [pre_pro.truncate_pad(line, num_steps) for line in target]
    source = [source_vocab[line] for line in source]
    target = [target_vocab[line] for line in target]
    return source, target



class translation_dastaset(Dataset):
    def __init__(self, source, target):
        super().__init__()
        self.source = source
        self.target = target
    
    def __len__(self):
        return len(self.source)
    
    def __getitem__(self, index):
        return torch.tensor(self.source[index]), torch.tensor(self.target[index])


class Encoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Linear(vocab_size, embed_size)
        self.gru = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    
    def forward(self, x):  # x:[L, B, V]
        embed_x = self.embedding(x)  # [L, B, V]->[L, B, E]
        outputs, state = self.gru(embed_x)  # state:[num_layers, B, H]
        return outputs, state


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Linear(vocab_size, embed_size)
        self.gru = nn.GRU(input_size=embed_size + hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
    

    def get_init_state(self, enc_outputs):
        return enc_outputs[1]
    
    def forward(self, x, context, state):  # state:gru的上个时刻隐藏状态，context:encoder输出的上下文向量
        embed_x = self.embedding(x)
        dec_state = state
        c = context.repeat(embed_x.shape[0], 1, 1)  # context:[B, H]
        embed_x = torch.cat((embed_x, c), dim=2)  # [L, B, E]->[L, B, E+H]
        outputs, dec_state = self.gru(embed_x, dec_state)  # [L, B, H], [num_layers, B, H]
        outputs = self.output_layer(outputs).transpose(0, 1)  # [L, B, V]->[B, L, V]
        return outputs, dec_state 


class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, num_layers, hidden_size, source_vocab_size, target_vocab_size, dropout):
        super().__init__()
        self.encoder = Encoder(embed_size, hidden_size, source_vocab_size, num_layers, dropout)
        self.decoder = Decoder(embed_size, hidden_size, target_vocab_size, num_layers, dropout)
    
    def forward(self, x, y):  # x/y:[L, B, V]
        enc_outputs = self.encoder(x)
        state = self.decoder.get_init_state(enc_outputs)
        return self.decoder(y, context=state[-1], state=state)


class maskedCELoss(nn.CrossEntropyLoss):
    def _seqMask(self, seq, seq_len):  # seq:[B, L], seq_len:[B]
        pos = torch.arange(seq.shape[1]).repeat(seq.shape[0], 1).to(device)
        mask = pos < seq_len.unsqueeze(1).repeat(1, seq.shape[1])
        seq[~mask] = 0
        return seq
    
    def forward(self, y_hat, y, y_length):
        self.reduction = 'none'
        weight = torch.ones(y.shape).to(device)
        weight = self._seqMask(weight, y_length)
        y_hat = y_hat.transpose(1, 2)  # [B, V, L]
        ce_loss = super(maskedCELoss, self).forward(y_hat, y)
        masked_loss = (weight * ce_loss)  # [B, L]
        return masked_loss



def train_epoch(net, date_iter, optimizer, loss_fn, source_vocab, target_vocab, max_norm):
    net.train()
    total_loss = 0.
    total_words = 0.
    for x, y in date_iter:  # x, y:[B, L]
        with torch.no_grad():
            y_length = torch.sum(y != target_vocab['<pad>'], dim=1).to(device)  # [B]
        x = F.one_hot(x.T, len(source_vocab)).float().to(device)
        y = y.to(device)
        bos = torch.tensor([target_vocab['<bos>']] * y.shape[0], dtype=torch.long, device=device)
        dec_input = torch.cat((bos.unsqueeze(1), y[:, :-1]), dim=1)
        dec_input = F.one_hot(dec_input.T, len(target_vocab)).float()  # [L, B, V]
        y_hat, _ = net(x, dec_input)
        loss = loss_fn(y_hat, y, y_length)
        with torch.no_grad():
            total_loss += loss.detach().clone().sum().item()
            total_words += y_length.sum().item()
        optimizer.zero_grad()
        loss.mean().backward()
        clip_grad_norm_(net.parameters(), max_norm=max_norm)
        optimizer.step()
    return total_loss / total_words
        

def train(net, num_epochs, data_iter, source_vocab, target_vocab, max_norm, lr, print_every=1):
    optimzer = optim.Adam(net.parameters(), lr=lr)
    loss_fn = maskedCELoss()
    for epoch in range(num_epochs):
        loss_per_word = train_epoch(net, data_iter, optimzer, loss_fn, source_vocab, target_vocab, max_norm)
        if (epoch + 1) % print_every == 0:
            print(f'epoch {epoch + 1}, loss_per_word {loss_per_word:.8f}')

def xavier_init_weight(x):
    if type(x) == nn.Linear:
        nn.init.xavier_uniform_(x.weight)
    elif type(x) == nn.GRU:
        for param in x._flat_weights_names:
            if 'weight' in param:
                nn.init.xavier_uniform_(x._parameters[param])


def predict(net, src, source_vocab, target_vocab, max_len):
    net.eval()
    outputs = []
    src = src.lower().split()
    src = src + ['<eos>']
    src_tokens = torch.tensor([source_vocab[src]], device=device, dtype=torch.long)  # [1, L]
    enc_input = F.one_hot(src_tokens.T, len(source_vocab)).float()  # [L, 1, V]
    _, enc_output = net.encoder(enc_input)
    context = enc_output[-1]  # [1, H]
    state = enc_output  # [num_layers, 1, H]
    dec_input = torch.tensor([[target_vocab['<bos>']]], device=device)  # [1, 1]
    for _ in range(max_len):
        dec_input = F.one_hot(dec_input, len(target_vocab)).float()  # [1, 1, V]
        dec_output, state = net.decoder(dec_input, context, state)  # [B, L, V], [num_layers, 1, H]
        dec_input = dec_output.argmax(dim=-1)  # [B, L]
        if dec_input.item() == target_vocab['<eos>']:
            break
        outputs.append(dec_input.item())
    return target_vocab.to_tokens(outputs)

    
def bleu(label, predict, k):
    label = label.split()
    predict = predict.split()
    score = math.exp(min(0., len(label) / len(predict)))
    for n in range(1, 1 + k):
        n_gram = {}
        match = 0.
        for i in range(0, len(predict) - n + 1):
            seq = ''.join(predict[i: i + n])
            n_gram[seq] = n_gram.get(seq, 0) + 1
        for i in range(0, len(label) - n + 1):
            seq = ''.join(label[i: i + n])
            if seq in n_gram and n_gram[seq] > 0:
                match += 1
                n_gram[seq] -= 1
        score *= math.pow(match / (len(predict) - n + 1), math.pow(1 / 2, n))
    return score


if __name__ == '__main__':
    num_examples = 1200
    batch_size = 256
    num_steps = 10
    lr = 0.005
    max_norm = 1.
    dropout = 0.1
    num_layers = 2
    hidden_size, embed_size = 32, 32
    num_epochs = 300


    pre_pro = rowDataPreProcess()
    lines = pre_pro.readFile(file_path)
    lines = [pre_pro.addSpace(line) for line in lines]
    source, target = pre_pro.tokenize(lines, num_examples)
    source_vocab = Vocab(source, min_freqs=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    target_vocab = Vocab(target, min_freqs=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    source, target = build_data(pre_pro, source, target, source_vocab, target_vocab, num_steps)
    

    train_data = DataLoader(translation_dastaset(source, target), batch_size=batch_size, shuffle=True, num_workers=4)
    
    net = EncoderDecoder(embed_size, num_layers, hidden_size, len(source_vocab), len(target_vocab), dropout)
    net.apply(xavier_init_weight)
    net = net.to(device)


    train(net, num_epochs, train_data, source_vocab, target_vocab, max_norm, lr)

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .', 'run !']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .', 'prenez vos jambes à vos cous !']
    for i in range(len(engs)):
        pred = predict(net, engs[i], source_vocab, target_vocab, num_steps)
        print(engs[i], '-->', pred, '-->bleu2: ', bleu(fras[i], pred, k=2))
    