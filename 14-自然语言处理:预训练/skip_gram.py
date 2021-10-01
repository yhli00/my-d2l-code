# 训练数据集：'ptb.train.txt'
# 去除出现次数少于10次的单词
# 使用下采样，单词的保留概率为sqrt(1e-4 / 单词出现的频率)
# 最大上下文窗口为5，负采样单词数为5
# 每个单词的负采样概率为 单词出现次数 ** 0.75 / sum(单词出现次数  ** 0.75)
# 使用二元交叉熵损失函数
# optimizer = adam
# lr = 0.002
# num_epochs = 5
# batch_size = 512
# embed_size = 100



from tqdm import tqdm
import random
import torch
import math
from torch import nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import torch.nn.functional as F


torch.manual_seed(123456)
file_path = '../data/ptb/ptb.train.txt'
device = torch.device('cuda')


class PreProcess():
    def __init__(self):
        pass

    def read_file(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        lines = [line.strip().lower() for line in lines]
        lines = [line.split() for line in lines]
        return lines


class Vocab:
    def __init__(self, lines, min_freqs=-1):
        self.tokens = [token for line in lines for token in line]
        counter = Counter(self.tokens)
        counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.token_to_idx = {'<unk>': 0}
        self.idx_to_token = ['<unk>']
        self.token_freq = [0]
        self.unk = 0
        for t, f in counter:
            if f >= min_freqs:
                self.idx_to_token.append(t)
                self.token_freq.append(f)
                self.token_to_idx[t] = len(self.idx_to_token) - 1
    
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, list):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        if not isinstance(indices, list):
            return self.idx_to_token[indices]
        return ' '.join(self.idx_to_token[indic] for indic in indices)


def sub_sample(lines, vocab):
    def _keep(token_freq, total_freq):
        if random.uniform(0, 1) < math.sqrt((1e-4 / (token_freq / total_freq))):
            return True
        return False
    total_freq = sum(vocab.token_freq)
    lines = [vocab[line] for line in lines]
    lines = [[token for token in line if token != vocab.unk] for line in lines]
    lines = [[token for token in line if _keep(vocab.token_freq[token], total_freq)] for line in lines]
    return lines


def get_centers_contexts(corpus, max_window_size):
    centers = []
    contexts = []
    for tokens in tqdm(corpus, desc='get_centers_contexts'):
        if len(tokens) < 2:
            continue
        window_size = random.randint(1, max_window_size)
        centers += tokens
        for i in range(len(tokens)):
            indices = list(range(
                max(0, i - window_size),
                min(i + window_size, len(tokens))
            ))
            indices.remove(i)
            context = [tokens[indic] for indic in indices]
            contexts.append(context)
    return centers, contexts


class RandomChoice:
    def __init__(self, weights, k=10000):
        self.i = 0
        self.weights = weights
        self.k = k
        self.samples = random.choices(list(range(1, len(weights) + 1)), weights=self.weights, k=self.k)
    
    def generate(self):
        if self.i >= self.k:
            self.i = 0
            self.samples = random.choices(list(range(1, len(self.weights) + 1)), weights=self.weights, k=self.k)
        sample = self.samples[self.i]
        self.i += 1
        return sample



def negative_sample(contexts, vocab, num_negatives):
    negative_samples = []
    weights = [freq ** 0.75 for freq in vocab.token_freq[1:]]
    random_choice = RandomChoice(weights, k=10000)
    for context in tqdm(contexts, desc='negative_sample'):
        k = 0
        negative = []
        while k < num_negatives:
            sample = random_choice.generate()
            if sample not in context:
                negative.append(sample)
                k += 1
        negative_samples.append(negative)
    return negative_samples
    

def collate_fn(data):  # [B, 1], [B, L2], [B, K]
    centers = [[ce] for ce, _, _ in data]
    contexts = [con for _, con, _ in data]
    negatives = [ne for _, _, ne in data]
    batch_size = len(centers)
    max_len = max([len(negatives[i]) + len(contexts[i]) for i in range(batch_size)])
    contexts_negatives = [contexts[i] + negatives[i] for i in range(batch_size)]
    labels = [[1] * len(contexts[i]) + [0] * (max_len - len(contexts[i])) for i in range(batch_size)]
    mask = [[1] * len(c_n) + [0] * (max_len - len(c_n)) for c_n in contexts_negatives]
    contexts_negatives = [c_n + [0] * (max_len - len(c_n)) for c_n in contexts_negatives]
    return (
        torch.tensor(centers, dtype=torch.long),  # [B, 1]
        torch.tensor(contexts_negatives, dtype=torch.long),  # [B, max_len]
        torch.tensor(labels, dtype=torch.float),  # [B, max_len]
        torch.tensor(mask, dtype=torch.float)  # [B, max_len]
    )


class PTBDataSet(Dataset):
    def __init__(self, centers, contexts, negatives):
        super().__init__()
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives
    
    def __len__(self):
        return len(self.centers)
    
    def __getitem__(self, index):
        return self.centers[index], self.contexts[index], self.negatives[index]


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.center_embedding = nn.Embedding(vocab_size, embed_size)
        self.context_embedding = nn.Embedding(vocab_size, embed_size)
    
    def forward(self, center, context_negative):  # [B, 1], [B, max_len]
        center_embed = self.center_embedding(center)
        context_negative_embed = self.context_embedding(context_negative)
        return torch.bmm(center_embed, context_negative_embed.transpose(1, 2)).squeeze(1)


def train_epoch(data_iter, net, optimizer, epoch_no):
    net.train()
    total_loss = 0.
    total_words = 0.
    for (centers, contexts_negatives, labels, masks) in tqdm(data_iter, desc=f'train epoch {epoch_no}'):
        centers = centers.to(device)
        contexts_negatives = contexts_negatives.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        y_hat = net(centers, contexts_negatives)
        loss = F.binary_cross_entropy_with_logits(y_hat.float(), labels, weight=masks, reduction='none')
        with torch.no_grad():
            total_loss += loss.clone().detach().sum().item()
            total_words += masks.clone().detach().sum().item()
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()
    return total_loss / total_words


def train(net, lr, num_epochs, data_iter, print_every=1):
    def weight_init(x):
        if type(x) == nn.Embedding:
            nn.init.xavier_uniform_(x.weight)
    net.apply(weight_init)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        loss_per_words = train_epoch(data_iter, net, optimizer, epoch + 1)
        if (epoch + 1) % print_every == 0:
            print(f'train epoch {epoch + 1}, loss_per_word {loss_per_words:.5f}')

def get_similar_word(word, embedding, vocab, k):
    embedding.eval()
    with torch.no_grad():
        word_id = vocab[word]
        weight = embedding.weight.data
        word_emded = weight[word_id]
        cos_sim = torch.mv(weight, word_emded) / (torch.sum(word_emded * word_emded) * torch.sum(weight * weight, dim=1))
        _, indices = torch.topk(cos_sim, k + 1)
        for i in range(0, k + 1):
            print(f'cos_sim {cos_sim[indices[i]]:.5f}, word \'{vocab.to_tokens(indices[i])}\'')





if __name__ == '__main__':
    min_freqs = 10
    max_window_size = 5
    num_negatives = 5
    batch_size = 512
    embed_size = 100
    num_epochs = 30
    lr = 0.002


    pre_process = PreProcess()
    lines = pre_process.read_file(file_path)
    vocab = Vocab(lines, min_freqs=10)
    print(len(vocab))
    corpus = sub_sample(lines, vocab)
    centers, contexts = get_centers_contexts(corpus, max_window_size)
    negatives = negative_sample(contexts, vocab, num_negatives)
    train_data = DataLoader(PTBDataSet(centers, contexts, negatives), shuffle=True, batch_size=batch_size, 
                            num_workers=4, collate_fn=collate_fn)


    net = SkipGram(len(vocab), embed_size)
    net = net.to(device)


    train(net, lr, num_epochs, train_data, print_every=1)


    get_similar_word('chip', net.center_embedding, vocab, 3)