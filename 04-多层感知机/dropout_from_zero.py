'''
隐藏层h1=256, h2=256
丢弃概率dropout1=0.2, dropout2=0.5
'''
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim

def dropout(x, p_drop):  # p_drop是丢弃概率
    if p_drop == 0:  # 全部保留
        return x
    if p_drop == 1:
        return torch.zeros(x.shape, dtype=x.dtype)
    mask = (torch.randn(x.shape).uniform_(0, 1) > p_drop).float()
    return x * mask / (1.0 - p_drop)

def evaluate(net, test_data):
    correct = 0.
    total_num = 0.
    for x, y in test_data:
        y_hat = net(x.reshape(-1, 784))
        y_hat = torch.argmax(y_hat, dim=1)
        correct += torch.sum(y_hat.type(y.dtype) == y).float().item()
        total_num += len(y)
    return correct / total_num


class mlp(nn.Module):
    def __init__(self, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.lin1 = nn.Linear(784, 256)
        self.lin2 = nn.Linear(256, 256)
        self.lin3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        h1 = self.relu(self.lin1(x))
        if self.is_train:  # 只有当训练时才采用dropout
            h1 = dropout(h1, 0.2)
        h2 = self.relu(self.lin2(h1))
        if self.is_train:
            h2 = dropout(h2, 0.5)
        return self.lin3(h2)

def init_weight(x):
    if type(x) == nn.Linear:
        nn.init.normal_(x.weight, mean=0, std=0.1)
        nn.init.zeros_(x.bias)


train_data = torchvision.datasets.FashionMNIST(root='../data', download=True, train=True, transform=transforms.ToTensor())
test_data = torchvision.datasets.FashionMNIST(root='../data', download=True, train=False, transform=transforms.ToTensor())
train_data = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=4)
test_data = DataLoader(test_data, batch_size=256, num_workers=4, shuffle=True)

net = mlp()
net.apply(init_weight)

num_epochs, lr = 10, 0.1
optimizer = optim.SGD(net.parameters(), lr=lr)
ls = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    total_loss = 0.
    total_num = 0.
    for x, y in train_data:
        y_hat = net(x.reshape(-1, 784))
        loss = ls(y_hat, y)
        total_loss += loss.detach().clone().item()
        total_num += len(y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        acc = evaluate(net, test_data)
        print(f'epoch {epoch + 1}, loss {(total_loss / total_num):.6f}, acc {acc * 100:.4f}%')
