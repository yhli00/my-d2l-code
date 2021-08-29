# resnet18的pytorch实现
# resnet中的残差块（包括两个卷积层，每个卷积层的输出通道数相同）：
#     + 一个padding=1的3*3卷积层（stride由参数指定），后面跟着BatchNorm和ReLU激活函数
#     + 一个padding=1、stride=1的3*3卷积层（输出通道数与第一个卷积层的输出通道数相同），后面跟着BatchNorm层
#     + 一个1*1的卷积层（stride由参数指定），得到的结果与上面两个卷积层的结果相加，然后经过ReLU激活函数得到残差块的输出
# resnet18由3个部分组成（卷积层、4个残差部分、线性层）：
#     + 卷积层：
#         - 输出通道为64、padding=3、stride=2的7*7卷积核
#         - 批量归一化层、ReLU激活函数
#         - stride=2、padding=1的3*3最大池化层
#     + 4个残差部分
#         -   1、第一个残差块：输出通道数为64，stride=1
#             2、第二个残差块：输出通道为64，stride=1
#         -   1、第一个残差块：输出通道数为128，stride=2
#             2、第二个残差块：输出通道为128，stride=1
#         -   1、第一个残差块：输出通道数为256，stride=2
#             2、第二个残差块：输出通道为128，stride=1
#         -   1、第一个残差块：输出通道数为512，stride=2
#             2、第二个残差块：输出通道为512，stride=1
#     + 一个平均池化层，使输出的图片大小为1*1
#     + 一个线性层，输出特征数为10
# 输入图片大小为96*96
# lr=0.05
# test acc=90.82%


import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader


torch.manual_seed(99999)
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


class Residual(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel)
        )
        if stride > 1:
            self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride)
        self.relu = nn.ReLU()
        self.stride = stride

    def forward(self, x):
        conv1 = self.conv1(x)
        if self.stride == 1:
            return self.relu(conv1 + x)
        else:
            return self.relu(conv1 + self.conv2(x))


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, padding=3, stride=2),  # [B, 1, 96, 96]->[B, 64, 48, 48]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2)  # [B, 64, 48, 48]->[B, 64, 24, 24]
        )
        self.conv2 = self.resnetBlock(64, 64, 2, stride=1, first_block=True)  # [B, 64, 24, 24]->[B, 64, 24, 24]
        self.conv3 = self.resnetBlock(64, 128, 2, 2, False)  # [B, 64, 24, 24]->[B, 128, 12, 12]
        self.conv4 = self.resnetBlock(128, 256, 2, 2, False)  # [B, 128, 12, 12]->[B, 256, 6, 6]
        self.conv5 = self.resnetBlock(256, 512, 2, 2, False)  # [B, 256, 6, 6]->[B, 256, 3, 3]
        self.conv = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4, 
            self.conv5,
            nn.AvgPool2d(kernel_size=3, stride=1)
        )
        self.lin = nn.Linear(512, 10)
    
    def forward(self, x):
        conv = self.conv(x)
        return self.lin(conv.reshape(-1, 512))

    
    def resnetBlock(self, in_channel, out_channel, num_layers, stride, first_block=False):
        layers = []
        for i in range(num_layers):
            if i == 0 and not first_block:
                layers.append(Residual(in_channel, out_channel, stride=stride))
            else:
                layers.append(Residual(out_channel, out_channel))
        return nn.Sequential(*layers)

def init_weight(x):
    if type(x) == nn.Conv2d or type(x) == nn.Linear:
        nn.init.xavier_uniform_(x.weight)


net = ResNet18()
net.apply(init_weight)
net = net.to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)


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