'''
num_epochs:10
lr=0.1
网络：784->256->10
optimizer:SGD
activate_fn:relu
'''
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


def relu(x):
    zeros = torch.zeros(x.shape, dtype=x.dtype)
    return torch.max(zeros, x)

def mlp(x):
    h1 = relu(torch.matmul(x.reshape(-1, 784), w1) + b1)
    return torch.matmul(h1, w2) + b2

def evaluate(net, test_data):
    correct = 0.
    total_num = 0
    for x, y in test_data:
        y_hat = net(x)
        y_hat = torch.argmax(y_hat, dim=1)
        correct += torch.sum(y_hat.type(y.dtype) == y).item()
        total_num += len(x)
    return correct / total_num


w1 = nn.Parameter(torch.normal(0, 0.01, (784, 256), requires_grad=True))
b1 = nn.Parameter(torch.tensor(0, dtype=torch.float32, requires_grad=True))
w2 = nn.Parameter(torch.normal(0, 0.1, (256, 10), requires_grad=True))
b2 = nn.Parameter(torch.tensor(0, dtype=torch.float32, requires_grad=True))
param = [w1, b1, w2, b2]


train_data = torchvision.datasets.FashionMNIST(root='../data', download=True, train=True, transform=transforms.ToTensor())
test_data = torchvision.datasets.FashionMNIST(root='../data', download=True, train=False, transform=transforms.ToTensor())
train_data = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=4)
test_data = DataLoader(test_data, batch_size=256, shuffle=True, num_workers=4)

num_epochs, lr = 10, 0.1
optimizer = torch.optim.SGD(params=param, lr=lr)
net = mlp
ls = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    total_loss = 0.
    total_num = 0
    for x, y in train_data:
        y_hat = net(x)
        loss = ls(y_hat, y)
        total_loss += loss.detach().clone() * len(x)
        total_num += len(x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        acc = evaluate(net, test_data)
        print(f'epoch {epoch + 1}, loss {(total_loss / total_num):.6f}, test_acc {acc * 100:.4f}%')

