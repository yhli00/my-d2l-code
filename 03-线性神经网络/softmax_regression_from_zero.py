# batch_size=256
# w由均值0方差0.01的正太分布生成
# b初始化为0
# lr=0.1

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

def softmax(x):
    exp_x = torch.exp(x)
    x_sum = exp_x.sum(dim=1, keepdim=True)
    return exp_x / x_sum

def softmax_regression(w, b, x):
    return softmax(torch.matmul(x.reshape(-1, 784), w) + b)

def sgd(params, lr):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad
            param.grad.zero_()

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[list(range(len(y))), y])

def evaluate(net, test_data, w, b):
    with torch.no_grad():
        correct = 0.
        total_nums = 0.
        for x, y in test_data:
            y_hat = net(w, b, x)
            labels_hat = y_hat.argmax(dim=1)
            labels_hat.type(y.dtype)
            correct += torch.sum(labels_hat == y).float()
            total_nums += len(y)
    return (correct / total_nums).item()


train_data = torchvision.datasets.FashionMNIST(root='../data', transform=transforms.ToTensor(), download=True, train=True)
test_data = torchvision.datasets.FashionMNIST(root='../data', transform=transforms.ToTensor(), download=True, train=False)
train_data = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=4)
test_data = DataLoader(test_data, batch_size=256, shuffle=True, num_workers=4)
w = torch.normal(0, 0.1, (784, 10), requires_grad=True)
b = torch.tensor(0., requires_grad=True)
net = softmax_regression
num_epochs = 10
lr = 0.1
test_acc = evaluate(net, test_data, w, b)
print(f'test_acc {test_acc:.4f}')
for epoch in range(num_epochs):
    total_loss = torch.tensor([])
    for x, y in train_data:
        y_hat = net(w, b, x)
        loss = cross_entropy(y_hat, y)
        loss.mean().backward()
        sgd([w, b], lr)
        total_loss = torch.cat((total_loss, loss.detach().clone()), dim=0)
    with torch.no_grad():
        test_acc = evaluate(net, test_data, w, b)
        avg_loss = torch.sum(total_loss) / len(total_loss)
        print(f'epoch {epoch + 1},  loss {avg_loss.item():.6f},  test_acc {test_acc:.4f}')
