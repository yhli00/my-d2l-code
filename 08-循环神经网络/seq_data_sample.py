# 实现顺序数据的随机采样和顺序采样（相邻的batch数据在空间上也是相邻的）

import random
import torch


# 随机采样
def seq_data_iter_random(corpus, batch_size, num_step):
    corpus = corpus[random.randint(0, num_step - 1):]
    num_seq = (len(corpus) - 1) // num_step
    initial_indics = list(range(0, num_seq * num_step, num_step))
    random.shuffle(initial_indics)
    num_batch = num_seq // batch_size

    def data(pos):
        return corpus[pos: pos + num_step]
    for i in range(0, num_batch * batch_size, batch_size):
        indics = initial_indics[i: i + batch_size]
        x = [data(j) for j in indics]
        y = [data(j + 1) for j in indics]
        yield torch.tensor(x), torch.tensor(y)


# 顺序采样
def seq_data_iter_sequential(corpus, batch_size, num_step):
    corpus = corpus[random.randint(0, num_step - 1):]
    num_tokens = (len(corpus) - 1) // batch_size * batch_size
    x = torch.tensor(corpus[:num_tokens])
    y = torch.tensor(corpus[1:num_tokens + 1])
    x = x.reshape(batch_size, -1)
    y = y.reshape(batch_size, -1)
    num_seq = x.shape[1] // num_step
    for i in range(0, num_seq * num_step, num_step):
        yield x[:, i:i + num_step], y[:, i:i + num_step]



if __name__ == '__main__':
    x = range(35)
    for i, j in seq_data_iter_sequential(x, batch_size=2, num_step=5):
        print('X:', i)
        print('Y:', j)