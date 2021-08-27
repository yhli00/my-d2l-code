# NiN的pytorch实现
# NiN由4个NiN块组成，其中不包括任何全连接层：
#     + 第一个NiN块，由3个部分组成，后面接着一个核大小为3、步幅为2的最大池化层：
#         - 输出通道为96，卷积核大小为11、步幅为4、padding为0的卷积核
#         - 两个核大小为1的卷积层
#         - ReLU激活函数
#     + 第二个NiN块，后面接一个核大小为3、步幅为2的最大池化层：
#         - 输出通道为256、卷积核大小为5、步幅为1、padding为2的卷积层
#         - 两个核大小为1的卷积核
#     + 第三个NiN块，后面接一个核大小为3、步幅为2的最大池化层，然后接着一个丢弃概率为0.5的Dropout层：
#         - 输出通道为384、卷积核大小为3、步幅为1、padding为1的卷积层
#         - 两个核大小为1的卷积层
#     + 第四个NiN块，后面接一个平均池化层，使得输出的特征大小为1*1
#         - 输出通道为10，卷积核大小为3、步幅为1、padding为1的卷积层
#         - 两个核大小为1的卷积层
# lr = 0.1
# optimizer = SGD
# num_epochs = 10
# test acc=84.45%
# 疑问：为啥训练时的test acc会很不稳定？

import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


torch.manual_seed(99999)
device = torch.device('cuda')


train_data = torchvision.datasets.FashionMNIST(
    root='../data',
    train=True,
    download=True,
    transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
)
test_data = torchvision.datasets.FashionMNIST(
    root='../data',
    train=False,
    download=True,
    transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
)
train_data = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=4)
test_data = DataLoader(test_data, batch_size=256, shuffle=True, num_workers=4)


class NiN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            self._NiNBlock(1, 96, kernel_size=11, stride=4, padding=0),  # [B, 1, 224, 224]->[B, 96, 54, 54]
            nn.MaxPool2d(kernel_size=3, stride=2),  # [B, 96, 54, 54]->[B, 96, 26, 26]
            self._NiNBlock(96, 256, kernel_size=5, stride=1, padding=2),  # [B, 96, 26, 26]->[B, 256, 26, 26]
            nn.MaxPool2d(kernel_size=3, stride=2),  # [B, 256, 26, 26]->[B, 256, 12, 12]
            self._NiNBlock(256, 384, kernel_size=3, stride=1, padding=1),  # [B, 256, 12, 12]->[B, 384, 12, 12]
            nn.MaxPool2d(kernel_size=3, stride=2),  # [B, 384, 12, 12]->[B, 384, 5, 5]
            nn.Dropout(p=0.5),
            self._NiNBlock(384, 10, kernel_size=3, stride=1, padding=1),  # [B, 384, 5, 5]->[B, 10, 5, 5]
            nn.AvgPool2d(kernel_size=5)
        )

    def forward(self, x):
        conv = self.conv(x)  # [B, 10, 1, 1]
        return conv.reshape(-1, 10)

    def _NiNBlock(self, in_channel, out_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=1),
            nn.ReLU()
        )


def init_weight(x):
    if type(x) == nn.Conv2d or type(x) == nn.Linear:
        nn.init.xavier_uniform_(x.weight)


net = NiN()
net.apply(init_weight)
net = net.to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)


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


optimizer.zero_grad()
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