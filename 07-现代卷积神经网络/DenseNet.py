# DenseNet的pytorch实现
# Densnet稠密块，由n个输出通道数相同的卷积块组成，每个卷积块由padding=1的3*3卷积层、BatchNorm、ReLU组成，稠密块把这n个卷积块
# 的输出在输出通道这一维度拼接起来作为最后的输出
# Densnet过渡块：由给定输出通道数的1*1卷积层、stride=2的2*2平均池化层组成
# DenseNet由一个卷积层、4个稠密块（每个稠密块包含3个卷积块）、每两个稠密块之间有一个过渡块（一共3个）、一个线性输出层组成：
#     + 第一个卷积层：
#         - 输出通道为64、padding=3、stride=2的7*7卷积核
#         - 批量归一化层、ReLU激活函数
#         - stride=2、padding=1的3*3最大池化层
#     + 4个稠密块，每个稠密块由4个卷积块组成，每个卷积块的输出维度都是32，每两个稠密块之间是一个过渡块，用来将输出通道数减半
#     + 一个平均池化层将图片大小变为1*1
#     + 一个线性输出层
# lr=0.1
# 输入图片大小为96*96
# lr=0.1
# epoch 9 test acc=90.70%
# epoch 10 test acc=81.92%


import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader


torch.manual_seed(98765)
device = torch.device('cuda')


train_data = torchvision.datasets.FashionMNIST(
    root='../data',
    train=True,
    download=True,
    transform=transforms.Compose([transforms.Resize((96, 96)), transforms.ToTensor()])
)
test_data = torchvision.datasets.FashionMNIST(
    root='../data',
    train=False,
    download=True,
    transform=transforms.Compose([transforms.Resize((96, 96)), transforms.ToTensor()])
)
train_data = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=4)
test_data = DataLoader(test_data, batch_size=256, shuffle=True, num_workers=4)


class DenseNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._convBlock(in_channel + i * out_channel, out_channel))
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        for layer in self.conv:
            y = layer(x)
            x = torch.cat([x, y], dim=1)
        return x
    
    def _convBlock(self, in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )


class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, padding=3, stride=2),  # [B, 1, 96, 96]->[B, 64, 48, 48]
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.dense = self._dense_transition_block(4, 32)  # [B, 64, 48, 48]->[B, 96, 24, 24]->[B, 112, 12, 12]->[B, 120, 6, 6]->[B, 248, 6, 6]
        self.avg = nn.AvgPool2d(kernel_size=6, stride=1)
        self.lin = nn.Linear(248, 10)
    
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.dense(conv1)
        return self.lin(self.avg(conv2).reshape(-1, 248))

    def _dense_transition_block(self, num_dense, out_channel):
        in_channel = 64
        layers = []
        for i in range(num_dense):
            layers.append(DenseNetBlock(in_channel, out_channel, 4))
            in_channel += 4 * out_channel
            if i != num_dense - 1:
                layers.append(self._transitionBlock(in_channel, in_channel // 2))
            in_channel = in_channel // 2
        return nn.Sequential(*layers)

    def _transitionBlock(self, in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

def init_weight(x):
    if type(x) == nn.Conv2d or type(x) == nn.Linear:
        nn.init.xavier_uniform_(x.weight)


net = DenseNet()
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