# pytorch实现LeNet，LeNet每个卷积层或者全连接层之后、激活函数之前都跟着一个Batch_norm层:
#     1、卷积层（大小为5的卷积核，6个输出通道, padding=0）+ Sigmoid激活函数 + avg池化层（步长为2, 大小为2）
#     2、卷积层（大小为5的卷积核，16个输出通道, padding=0） + Sigmoid激活函数 + avg池化层（步长为2， 大小为2）
#     3、两个线性层（维度=120和84） + Sigmoid激活函数
#     4、输出层
# 图片输入大小为28*28
# lr=1.0
# test acc=86.17%

import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


torch.manual_seed(1234)
device = torch.device('cuda')


train_data = torchvision.datasets.FashionMNIST(
    root='../data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
test_data = torchvision.datasets.FashionMNIST(
    root='../data',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)
train_data = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=4)
test_data = DataLoader(test_data, batch_size=256, shuffle=True, num_workers=4)


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),  # [B, 1, 28, 28]->[B, 6, 24, 24]
            nn.BatchNorm2d(6),  # 参数是输出通道数目
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # [B, 6, 24, 24]->[B, 6, 12, 12]
            nn.Conv2d(6, 16, kernel_size=5),  # [B, 6, 12, 12]->[B, 16, 8, 8]
            nn.BatchNorm2d(16),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # [B, 16, 8, 8]->[B, 16, 4, 4]
        )
        self.lin = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.BatchNorm1d(120),  # 参数是输出特征数量
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        conv = self.conv(x)
        return self.lin(conv.reshape(-1, 16 * 4 * 4))


def init_weight(x):
    if type(x) == nn.Linear or type(x) == nn.Conv2d:
        nn.init.xavier_uniform_(x.weight)


net = LeNet()
net.apply(init_weight)
net = net.to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1.0)


def evaluate(net, test_data):
    net.eval()
    correct, total_num = 0., 0.
    for x, y in test_data:
        x = x.to(device)
        y = y.to(device)
        y_hat = net(x)
        y_hat = torch.argmax(y_hat, dim=1)
        correct += torch.sum(y_hat.type(y.dtype) == y).float().item()
        total_num += len(y)
    return correct / total_num


num_epochs = 10
for epoch in range(num_epochs):
    net.train()
    epoch_loss = 0.
    for x, y in train_data:
        x = x.to(device)
        y = y.to(device)
        y_hat = net(x)
        loss = loss_fn(y_hat, y)
        epoch_loss = loss.detach().clone().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        print(f'epoch {epoch + 1}, loss {epoch_loss:.6f}, acc {evaluate(net, test_data) * 100:.4f}%')