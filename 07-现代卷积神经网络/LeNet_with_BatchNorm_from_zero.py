# pytorch实现LeNet，LeNet每个卷积层或者全连接层之后、激活函数之前都跟着一个Batch_norm层:
#     1、卷积层（大小为5的卷积核，6个输出通道, padding=0）+ Sigmoid激活函数 + avg池化层（步长为2, 大小为2）
#     2、卷积层（大小为5的卷积核，16个输出通道, padding=0） + Sigmoid激活函数 + avg池化层（步长为2， 大小为2）
#     3、两个线性层（维度=120和84） + Sigmoid激活函数
#     4、输出层
# 图片输入大小为28*28
# lr=1.0
# test acc=85.55%

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


def batch_norm(x, gamma, beta, moving_mean, moving_var, eps, momentun):
    if not torch.is_grad_enabled():  # 在训练模式
        x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        if len(x.shape) == 2:  # 线性层
            mean = x.mean(dim=0, keepdim=True)
            var = ((x - mean)**2).mean(dim=0, keepdim=True)
        else:  # 卷积层
            mean = x.mean(dim=(0, 2, 3), keepdim=True)  # 在“通道”那个维度求均值
            var = ((x - mean)**2).mean(dim=(0, 2, 3), keepdim=True)
        moving_mean = (1 - momentun) * mean + momentun * moving_mean  # moving_mean用来估计整个训练集的均值
        moving_var = (1 - momentun) * var + momentun * moving_var
        x_hat = (x - mean) / torch.sqrt(var + eps)
    y = gamma * x_hat + beta
    return y, moving_mean, moving_var


class BatchNorm(nn.Module):
    def __init__(self, num_dims, num_features):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)
    
    def forward(self, x):
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)
        y, self.moving_mean, self.moving_var = batch_norm(x, self.gamma, self.beta, self.moving_mean, self.moving_var,
                                                          eps=1e-5, momentun=0.1)
        return y


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),  # [B, 1, 28, 28]->[B, 6, 24, 24]
            BatchNorm(4, 6),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # [B, 6, 24, 24]->[B, 6, 12, 12]
            nn.Conv2d(6, 16, kernel_size=5),  # [B, 6, 12, 12]->[B, 16, 8, 8]
            BatchNorm(4, 16),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # [B, 16, 8, 8]->[B, 16, 4, 4]
        )
        self.lin = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            BatchNorm(2, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            BatchNorm(2, 84),
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
