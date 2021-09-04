# 1、按行读取文件
# 2、实现分词器（单词级别和字符级别）
# 3、实现词表（字符级别）:
#     + 传入参数：min_freqs, tokens
#     + 步骤：
#         - Counter()计数，得到self.token_freqs(list[tuple])
#         - 构造self.tokens_to_idx(dict)、self.idx_to_tokens(list)
#         - 实现self.__len__、self.to_tokens(把索引转化为字符)、self.__getitem__(把字符转化成索引)


import re
from collections import Counter


file_name = '../data/timemachine.txt'


def read_txt(filename):
    with open(filename, 'r')as f:
        lines = f.readlines()
    return [re.sub('[^a-zA-Z]+', ' ', line).strip().lower() for line in lines]


def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]


class Vocab:
    def __init__(self, tokens, min_freqs=0):
        counter = Counter([token for line in tokens for token in line])
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.unk = 0
        self.idx_to_token, self.token_to_idx = ['<unk>'], {'<unk>': 0}
        for token, freqs in self.token_freqs:
            if freqs >= min_freqs and token not in self.idx_to_token:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    
    def __len__(self):
        return len(self.token_to_idx)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, list):
            return self.token_to_idx[tokens]
        return [self.token_to_idx[token] for token in tokens]
    
    def to_tokens(self, indics):
        if not isinstance(indics, list):
            return self.idx_to_token[indics]
        return [self.idx_to_token[indic] for indic in indics]


if __name__ == '__main__':
    lines = read_txt(file_name)
    tokens = tokenize(lines, 'char')
    corpus = [letter for token in tokens for letter in token]
    vocab = Vocab(corpus)
    print(len(vocab))
    print(len(corpus))
    for i in [10]:
        print(tokens[i])
        print(vocab[tokens[i]])
        print(vocab.to_tokens(vocab[tokens[i]]))
    print(vocab.token_freqs[:10])