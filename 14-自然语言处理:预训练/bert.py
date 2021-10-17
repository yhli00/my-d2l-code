# bert:
#     + bert的输入由三部分相加：
#         - 词嵌入
#         - 位置嵌入
#         - 端嵌入
#     三个部分都是由学习得到的
#     + bert完成的两个任务：
#         - 遮蔽语言模型：
#             · 随机选取15%的token作为预测的token
#             · 在这15%的token中，80%的概率被替换成一个特殊token'<mask>'，10%的概率替换成随机token，10%的概率不变
#         - 下一句预测：
#             · 判断输入的两个句子中，第二个句子是不是第一个句子的下一句
#             · 训练时50%的概率是下一句，50%的概率不是下一句

# 数据处理：
#     + 只保留至少有两句话的段落（每句话以“.”分隔，数据集中的每一行是一段）
#     + 使用以空格分割的分词，去除出现次数少于5次的单词
#     + 先进行两个句子的拼接，得到nsp数据，再进行mlm操作
#     + 两个句子拼接后，再加上特殊字符的最大长度是64
#     + 去掉拼接后长度大于max_len的句子


# 模型细节：
#     + mlm:在bert词嵌入后，经过一个线性层，后接relu，后接layer_norm，最后线性输出层
#     + nsp:在bert词嵌入后，经过一个线性层，后接tanh，最后线性输出层
#     + batch_size = 512
#     + max_len = 64
#     + 使用2层bert，128个隐藏单元，2个注意力头
#     + mlm和nsp的中间层维度是128
#     + positionWiseFFN的中间层维度是256
#     + nn.CrossEntropyLoss
#     + adam, lr=1e-3
#     + q_size, k_size, v_size = 128
#     + dropout=0.2



from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import random
from collections import Counter
import torch.nn.functional as F
import math
import copy


torch.manual_seed(1234)
device = torch.device('cuda:0')
file_name = '../data/wikitext-2/wiki.train.tokens'


class PreProcess:

    @staticmethod
    def read_file():
        with open(file_name, 'r') as f:
            lines = f.readlines()
        random.shuffle(lines)
        paragraphs = [para.lower().split(' . ') for para in lines if len(para.split(' . ')) >= 2]
        paragraphs = [[line.strip() for line in para] for para in paragraphs]
        paragraphs = [[line for line in para if len(line) != 0] for para in paragraphs]
        paragraphs = [[PreProcess._tokenize(line) for line in para] for para in paragraphs]
        return paragraphs

    @staticmethod
    def _tokenize(line):
        return line.split()


class Vocab:
    def __init__(self, paras, min_freq=-1, reserved_token=[]):
        self.corpus = [token for para in paras for line in para for token in line]
        counter = Counter(self.corpus)
        self.token_to_idx = {'<unk>': 0}
        self.idx_to_token = ['<unk>']
        self.unk = 0
        for token in reserved_token:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1
        for token, freq in counter.items():
            if freq >= min_freq:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, list):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        if not isinstance(indices, list):
            return self.idx_to_token[indices]
        return [self.idx_to_token[indic] for indic in indices]




class WikiDataset(Dataset):
    def __init__(self, max_len, vocab):
        super().__init__()
        (self.all_paded_mlm_tokens, self.all_paded_segments,
         self.all_valid_lens, self.all_paded_mlm_positions, self.all_paded_mlm_labels,
         self.all_mlm_weights, self.all_is_next) = self._get_all_nsp_and_mlm_data(max_len, vocab)


    def __len__(self):
        return len(self.all_paded_mlm_tokens)
    
    def __getitem__(self, index):
        return (
            torch.tensor(self.all_paded_mlm_tokens[index], dtype=torch.long),
            torch.tensor(self.all_paded_segments[index], dtype=torch.long),
            torch.tensor(self.all_valid_lens[index], dtype=torch.long),
            torch.tensor(self.all_paded_mlm_positions[index], dtype=torch.long),
            torch.tensor(self.all_paded_mlm_labels[index], dtype=torch.long),
            torch.tensor(self.all_mlm_weights[index], dtype=torch.float),
            torch.tensor(self.all_is_next[index], dtype=torch.long)
        )
    
    def _get_all_nsp_data(self, paras, max_len):
        all_tokens = []
        all_segments = []
        all_is_next = []
        for para in paras:
            for i in range(len(para) - 1):
                token_x, token_y, is_next = self._get_nsp_data_from_one_para(para[i], para[i + 1], paras)
                if len(token_x) + len(token_y) + 3 > max_len:
                    continue
                tokens_x_y, segment = self._get_merge_and_segment(token_x, token_y)
                all_tokens.append(tokens_x_y)
                all_segments.append(segment)
                all_is_next.append(int(is_next))
        return all_tokens, all_segments, all_is_next


    def _get_nsp_data_from_one_para(self, line1, line2, paras):
        if random.random() < 0.5:  # 50%的概率是下一句
            next_line = line2
            is_next = True
        else:
            next_line = random.choice(random.choice(paras))
            is_next = False
        return line1, next_line, is_next



    def _get_merge_and_segment(self, x, y):  # x, y是句子的list
        token_x = ['<cls>'] + x + ['<sep>']
        segment = [1] * len(token_x)
        token_y = y + ['<sep>']
        segment = segment + [0] * len(token_y)
        return token_x + token_y, segment


    def _get_mlm_data_from_one_line(self, tokens, all_mlm_positions, vocab):
        mlm_tokens = copy.deepcopy(tokens)
        random.shuffle(all_mlm_positions)
        pred_mlm_positions_and_labels = []
        for position in all_mlm_positions:
            if len(pred_mlm_positions_and_labels) >= max(1, round(0.15 * len(tokens))):
                break
            if random.random() < 0.8:
                mlm_tokens[position] = '<mask>'
            else:
                if random.random() < 0.5:
                    mlm_tokens[position] = tokens[position]
                else:
                    mlm_tokens[position] = vocab.to_tokens(random.randint(5, len(vocab) - 1))
            pred_mlm_positions_and_labels.append((position, tokens[position]))
        return mlm_tokens, pred_mlm_positions_and_labels


    def _get_all_mlm_data(self, lines, vocab):
        all_mlm_tokens = []
        all_pred_mlm_labels = []
        all_pred_mlm_positions = []
        for tokens in lines:
            pred_mlm_positions = []
            for i, token in enumerate(tokens):
                if token != '<cls>' and token != '<sep>':
                    pred_mlm_positions.append(i)
            mlm_tokens, pred_mlm_positions_and_labels = self._get_mlm_data_from_one_line(tokens, pred_mlm_positions, vocab)
            pred_mlm_positions_and_labels = sorted(pred_mlm_positions_and_labels, key=lambda x: x[0])
            all_mlm_tokens.append(vocab[mlm_tokens])
            mlm_positions = [x for x, _ in pred_mlm_positions_and_labels]
            mlm_labels = [y for _, y in pred_mlm_positions_and_labels]
            all_pred_mlm_positions.append(mlm_positions)
            all_pred_mlm_labels.append(vocab[mlm_labels])
        return all_mlm_tokens, all_pred_mlm_positions, all_pred_mlm_labels


    def _get_all_nsp_and_mlm_data(self, max_len, vocab):
        all_paded_mlm_tokens = []
        all_paded_segments = []
        all_paded_mlm_positions = []
        all_paded_mlm_labels = []
        all_valid_lens = []
        all_mlm_weights = []
        paras = PreProcess.read_file()
        max_mlm_len = max(1, round(max_len * 0.15))
        all_tokens, all_segments, all_is_next = self._get_all_nsp_data(paras, max_len)
        all_mlm_tokens, all_mlm_positins, all_mlm_labels = self._get_all_mlm_data(all_tokens, vocab)
        for i in range(len(all_mlm_tokens)):
            valid_len = len(all_mlm_tokens[i])
            valid_mlm_position_len = len(all_mlm_positins[i])
            all_paded_mlm_tokens.append(all_mlm_tokens[i] + [vocab['<pad>']] * (max_len - valid_len))
            all_paded_segments.append(all_segments[i] + [0] * (max_len - valid_len))
            all_paded_mlm_positions.append(all_mlm_positins[i] + [0] * (max_mlm_len - valid_mlm_position_len))
            all_paded_mlm_labels.append(all_mlm_labels[i] + [0] * (max_mlm_len - valid_mlm_position_len))
            all_valid_lens.append(valid_len)
            all_mlm_weights.append([1.] * valid_mlm_position_len + [0.] * (max_mlm_len - valid_mlm_position_len))
        return (
            all_paded_mlm_tokens,
            all_paded_segments,
            all_valid_lens,
            all_paded_mlm_positions,
            all_paded_mlm_labels,
            all_mlm_weights,
            all_is_next
        )


def seq_mask(x, valid_lens, value=0):  # [B, L], [B]
    pos = torch.arange(x.shape[1], device=device).repeat(x.shape[0], 1)
    mask = pos < valid_lens.reshape(-1, 1).repeat(1, x.shape[1])
    x[~mask] = value
    return x



def masked_softmax(x, valid_lens, value=-1e9):  # [B, num_heads, num_q, num_v], [B]
    batch_size, num_heads, num_q, num_v = x.shape
    x = x.reshape(-1, num_v)
    x_valid_lens = valid_lens.repeat_interleave(num_heads * num_q)
    x = seq_mask(x, x_valid_lens, value=value)
    x = x.reshape(batch_size, num_heads, num_q, num_v)
    return F.softmax(x, dim=-1)



class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, q, k, v, valid_lens):  # [B, num_heads, num_q, q_size], [B]
        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(q.shape[-1])  # [B, num_heads, num_q, num_k]
        attn = masked_softmax(attn, valid_lens)
        return torch.matmul(self.dropout(attn), v)  # [B, num_heads, num_q, v_size]


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, q_size, k_size, v_size, num_heads, dropout):
        super().__init__()
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(hidden_size, q_size * num_heads)
        self.W_k = nn.Linear(hidden_size, k_size * num_heads)
        self.W_v = nn.Linear(hidden_size, v_size * num_heads)
        self.W_o = nn.Linear(num_heads * v_size, hidden_size)
    
    def forward(self, q, k, v, k_valid_lens):  # [B, L, hidden_size], [B]
        q_size = q.shape[-1]
        v_size = v.shape[-1]
        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)
        q = q.reshape(q.shape[0], q.shape[1], -1, q_size)
        k = k.reshape(k.shape[0], k.shape[1], -1, q_size)
        v = v.reshape(v.shape[0], v.shape[1], -1, v_size)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [B, num_heads, num_q, q_size]
        output = self.attention(q, k, v, k_valid_lens)  # [B, num_heads, num_q, v_size]
        output = output.transpose(1, 2)
        return self.W_o(output.reshape(output.shape[0], output.shape[1], -1))  # [B, num_q, H]


class PositionWiseFFN(nn.Module):
    def __init__(self, hidden_size, ffn_hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size)
    
    def forward(self, x):  # [B, L, H]
        return self.fc2(self.relu(self.fc1(x)))


class AddNorm(nn.Module):
    def __init__(self, dropout, hidden_size):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x, y):
        return self.layer_norm(x + self.dropout(y))


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, q_size, k_size, v_size, num_heads, dropout, ffn_hidden_size):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, q_size, k_size, v_size, num_heads, dropout)
        self.add_norm1 = AddNorm(dropout, hidden_size)
        self.position_wise_ffn = PositionWiseFFN(hidden_size, ffn_hidden_size)
        self.add_norm2 = AddNorm(dropout, hidden_size)
    
    def forward(self, x, x_valid_lens):
        y1 = self.attention(x, x, x, x_valid_lens)
        y1 = self.add_norm1(x, y1)
        y2 = self.position_wise_ffn(y1)
        return self.add_norm2(y1, y2)


class BertEncoder(nn.Module):
    def __init__(self, num_layers, vocab_size, hidden_size, q_size, 
                 k_size, v_size, num_heads, dropout, ffn_hidden_size, max_len=1000):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.segment_embedding = nn.Embedding(2, hidden_size)
        self.position_embedding = nn.Parameter(torch.rand(1, max_len, hidden_size))
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                f'{i}',
                TransformerEncoder(hidden_size, q_size, k_size, v_size, num_heads, dropout, ffn_hidden_size)
            )
    
    def forward(self, x, x_segments, x_valid_lens):  # [B, L], [B]
        x_embedding = self.token_embedding(x) + self.segment_embedding(x_segments) + self.position_embedding[:, :x.shape[1], :]
        x_encode = x_embedding
        for blk in self.blks:
            x_encode = blk(x_encode, x_valid_lens)
        return x_encode


class MaskLM(nn.Module):
    def __init__(self, hidden_size, mlm_hidden_size, vocab_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlm_hidden_size),
            nn.ReLU(),
            nn.LayerNorm(mlm_hidden_size),
            nn.Linear(mlm_hidden_size, vocab_size)
        )
    
    def forward(self, x, pred_mlm_positions):  # x:[B, L, H], pred_mlm_positions:[B, L_2](L_2 < L)
        batch_size = x.shape[0]
        mlm_position_len = pred_mlm_positions.shape[1]
        batch = torch.arange(batch_size, dtype=torch.long, device=device)
        batch = batch.repeat_interleave(mlm_position_len)
        mlm_input = x[batch, pred_mlm_positions.reshape(-1), :]  # [B*L_2, H]
        return self.mlp(mlm_input.reshape(batch_size, mlm_position_len, -1))  # [B, L_2, V]


class NextSP(nn.Module):
    def __init__(self, hidden_size, nsp_hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, nsp_hidden_size),
            nn.Tanh(),
            nn.Linear(nsp_hidden_size, 2)
        )
    
    def forward(self, x):  # [B, L, H]
        return self.mlp(x[:, 0, :])  # [B, H]->[B, 2]


class BertModel(nn.Module):
    def __init__(self, num_layers, vocab_size, hidden_size, q_size, k_size, 
                 v_size, num_heads, dropout, ffn_hidden_size, mlm_hidden_size, nsp_hidden_size):
        super().__init__()
        self.bert = BertEncoder(num_layers, vocab_size, hidden_size, q_size, 
                                k_size, v_size, num_heads, dropout, ffn_hidden_size)
        self.mlm = MaskLM(hidden_size, mlm_hidden_size, vocab_size)
        self.nsp = NextSP(hidden_size, nsp_hidden_size)
    
    def forward(self, x, x_segments, x_valid_lens, pred_mlm_positions):  # [B, L], [B, L], [B], [B, L_2]
        x_encode = self.bert(x, x_segments, x_valid_lens)
        mlm_preds = self.mlm(x_encode, pred_mlm_positions)  # [B, L_2, V]
        nsp_preds = self.nsp(x_encode)  # [B, 2]
        return x_encode, mlm_preds, nsp_preds


def get_batch_loss(mlm_preds, nsp_preds, mlm_labels, mlm_weights, is_next):  # [B, L_2, V], [B, 2], [B, L_2], [B, L_2], [B]
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    mlm_loss = loss_fn(mlm_preds.reshape(-1, mlm_preds.shape[-1]), mlm_labels.reshape(-1))  # [B * L_2]
    mlm_loss_per_word = torch.sum(mlm_loss * mlm_weights.reshape(-1)) / torch.sum(mlm_weights)
    nsp_loss = loss_fn(nsp_preds, is_next)
    nsp_loss_per_sentence = nsp_loss.sum() / nsp_preds.shape[0]
    return mlm_loss_per_word, nsp_loss_per_sentence, mlm_loss_per_word + nsp_loss_per_sentence


def train_epoch(train_data, optimizer, net, epoch):
    total_nsp_loss = 0.
    total_mlm_loss = 0.
    total_nsp_num = 0.
    total_mlm_num = 0.
    for (x, x_segments, x_valid_lens, pred_mlm_positions,
         pred_mlm_labels, mlm_weights, is_nexts) in tqdm(train_data, desc=f'train epoch{epoch + 1}'):
        x = x.to(device)
        x_segments = x_segments.to(device)
        x_valid_lens = x_valid_lens.to(device)
        pred_mlm_positions = pred_mlm_positions.to(device)
        pred_mlm_labels = pred_mlm_labels.to(device)
        mlm_weights = mlm_weights.to(device)
        is_nexts = is_nexts.to(device)
        _, mlm_preds, nsp_preds = net(x, x_segments, x_valid_lens, pred_mlm_positions)
        mlm_loss_per_word, nsp_loss_per_sentence, total_loss = get_batch_loss(mlm_preds, nsp_preds, 
                                                                              pred_mlm_labels, mlm_weights, is_nexts)
        with torch.no_grad():
            total_nsp_loss += nsp_loss_per_sentence.detach().clone().item() * mlm_preds.shape[0]
            total_mlm_loss += mlm_loss_per_word.detach().clone().item() * mlm_weights.sum()
            total_nsp_num += mlm_preds.shape[0]
            total_mlm_num += mlm_weights.sum()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    return total_nsp_loss / total_nsp_num, total_mlm_loss / total_mlm_num


def train(num_epochs, net, train_data, lr, print_every=1):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        nsp_loss_per_sentence, mlm_loss_per_word = train_epoch(train_data, optimizer, net, epoch)
        if (epoch + 1) % print_every == 0:
            print(f'epoch {epoch + 1} : nsp_loss_per_sentence {nsp_loss_per_sentence:.5f}, mlm_loss_per_word {mlm_loss_per_word: .5f}')



if __name__ == '__main__':
    batch_size = 512
    max_len = 64
    lr = 1e-3
    ffn_hidden_size = 256
    hidden_size = 128
    nsp_hidden_size = 128
    mlm_hidden_size = 128
    num_heads = 2
    num_layers = 2
    num_epochs = 10
    dropout = 0.2


    paras = PreProcess.read_file()
    vocab = Vocab(paras, min_freq=5, reserved_token=['<pad>', '<mask>', '<cls>', '<sep>'])
    wiki_dataset = WikiDataset(max_len, vocab)
    train_data = DataLoader(wiki_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    
    bert_model = BertModel(num_layers, len(vocab), hidden_size, hidden_size, hidden_size, hidden_size,
                           num_heads, dropout, ffn_hidden_size, mlm_hidden_size, nsp_hidden_size)
    bert_model = bert_model.to(device)
    

    train(num_epochs, bert_model, train_data, lr)