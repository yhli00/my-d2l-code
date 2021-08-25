# pytorch实现LeNet:
#     1、卷积层（大小为5的卷积核，6个输出通道, padding=2）+ Sigmoid激活函数 + avg池化层（步长为2, 大小为2）
#     2、卷积层（大小为5的卷积核，16个输出通道, padding=0） + Sigmoid激活函数 + avg池化层（步长为2， 大小为2）
#     3、两个线性层（维度=120和84） + Sigmoid激活函数
#     4、输出层
# num_epochs=10
# SGD优化器
# lr=0.9
# 线性层和卷积层使用xavier参数初始化

import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


torch.manual_seed(1234)

train_data = torchvision.datasets.FashionMNIST(root='../data', transform=transforms.ToTensor(), train=True, download=True)
test_data = torchvision.datasets.FashionMNIST(root='../data', transform=transforms.ToTensor(), train=False, download=True)
train_data = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=4)
test_data = DataLoader(test_data, batch_size=256, shuffle=True, num_workers=4)

print(next(iter(train_data))[0].shape)
print(next(iter(train_data))[1].shape)

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2) 
        )
        self.lin = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
    
    def forward(self, x):
        conv = self.conv(x)  # conv:[B, 16, 5, 5]
        return self.lin(conv.reshape(-1, 16 * 5 * 5))


def init_xavier(x):
    if type(x) == nn.Linear or type(x) == nn.Conv2d:
        nn.init.xavier_uniform_(x.weight)  # 线性层的权重或者卷积核的权重


net = LeNet()
net.apply(init_xavier)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.9)


def evaluate(net, test_data):
    net.eval()
    correct = 0.
    total_num = 0.
    for x, y in test_data:
        y_hat = net(x)
        y_hat = torch.argmax(y_hat, dim=1)
        correct += torch.sum(y_hat.type(y.dtype) == y).item()
        total_num += len(x)
    return correct / total_num


num_epochs = 10
for epoch in range(num_epochs):
    epoch_loss = 0.
    net.train()
    for x, y in train_data:
        y_hat = net(x)
        loss = loss_fn(y_hat, y)
        epoch_loss = loss.detach().clone().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        print(f'epoch {epoch + 1}, loss {epoch_loss}, acc {evaluate(net, test_data) * 100:.4f}%')
