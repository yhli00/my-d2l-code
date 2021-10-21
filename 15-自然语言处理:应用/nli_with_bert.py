# max_len = 128
# batch_size = 512
# lr = 1e-4
# num_epochs = 5

# test_acc:90.11%

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertModel, BertTokenizer
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '2,1'



train_file = '../data/snli_1.0/snli_1.0_train.txt'
test_file = '../data/snli_1.0/snli_1.0_test.txt'
label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
checkpoint = 'bert-base-cased'


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
        lines = [(label_set[line[0]], PreProcess._add_space(line[1]), PreProcess._add_space(line[2])) for line in tqdm(lines, desc=desc)]
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


class nli_dataset(Dataset):
    def __init__(self, is_train=True):
        super().__init__()
        if is_train:
            self.labels, self.premises, self.hypothesises = PreProcess.read_file(is_train)
        else:
            self.labels, self.premises, self.hypothesises = PreProcess.read_file(is_train)
    
    def __getitem__(self, index):
        return (
            self.labels[index], self.premises[index], self.hypothesises[index]
        )
    
    def __len__(self):
        return len(self.labels)



class Collate_Fn:
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def collate_fn(self, x):
        labels = [i[0] for i in x]
        premises = [i[1] for i in x]
        hypothesises = [i[2] for i in x]
        return (
            torch.tensor(labels, dtype=torch.long), 
            self.tokenizer(premises, hypothesises, max_length=self.max_len, padding='max_length', 
                           truncation=True, return_tensors='pt')
        )



class BertClassifier(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.bert = BertModel.from_pretrained(checkpoint)
        self.linear = nn.Linear(hidden_size, 3)
    
    def forward(self, x):
        encoded_x = self.bert(**x)
        return self.linear(encoded_x.last_hidden_state[:, 0, :])



def train_epoch(net, train_data, optimizer, scheduler, loss_fn, epoch):
    total_loss = 0.
    total_num = 0.
    net.train()
    for label, inputs in tqdm(train_data, desc=f'train epoch {epoch + 1}'):
        label = label.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        y_hat = net(inputs)
        loss = loss_fn(y_hat, label)
        with torch.no_grad():
            total_loss += loss.clone().detach().item() * label.shape[0]
            total_num += label.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return total_loss / total_num


def test(net, test_data):
    total_num = 0.
    total_correct = 0.
    net.eval()
    for label, inputs in test_data:
        label = label.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        y_hat = net(inputs)
        y_hat = y_hat.argmax(dim=-1)
        with torch.no_grad():
            total_correct += torch.sum(y_hat == label)
            total_num += label.shape[0]
    return total_correct / total_num


def train(net, train_data, test_data, lr, num_epochs, print_every=1):
    optimizer = AdamW(net.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_epochs * len(train_data))
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    for epoch in range(num_epochs):
        loss = train_epoch(net, train_data, optimizer, scheduler, loss_fn, epoch)
        acc = test(net, test_data)
        if (epoch + 1) % print_every == 0:
            print(f'epoch {epoch + 1}, avg_loss {loss:.4f}, test_acc {acc*100:.2f}%')




if __name__ == '__main__':
    num_epochs = 5
    lr = 1e-4
    batch_size = 128
    max_len = 128
    hidden_size = 768


    tokenizer = BertTokenizer.from_pretrained(checkpoint)

    train_set = nli_dataset(True)
    test_set = nli_dataset(False)
    train_data = DataLoader(train_set, shuffle=True, collate_fn=Collate_Fn(tokenizer, max_len).collate_fn,
                            batch_size=batch_size, num_workers=4)
    test_data = DataLoader(test_set, shuffle=True, collate_fn=Collate_Fn(tokenizer, max_len).collate_fn,
                            batch_size=batch_size, num_workers=4)       


    net = BertClassifier(hidden_size)
    net = nn.DataParallel(net, device_ids=[0, 1])
    net = net.to(device)

    train(net, train_data, test_data, lr, num_epochs)
