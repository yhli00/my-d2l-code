# GoogLeNet的pytorch实现
# 输入图片的像素为96*96
# GoogLeNet由inception块、卷积层和全连接层组成
# 一个inception块由4条路径组成：
#     + 一个1*1的卷积层
#     + 一个1*1的卷积层、一个3*3且padding=1的卷积层
#     + 一个1*1的卷积层、一个5*5且padding=2的卷积层
#     + 一个3*3且padding=1的maxPooling层、一个1*1的卷积层
#     + 将4条路径得到的结果在channel这一维度拼接起来，得到inception块的输出
# GoogLeNet的组成：
#     1、7*7、stride=2、padding=3、输出通道为64的卷积层，3*3、stride=2、padding=1的MaxPooling层
#     2、输出通道为64的1*1卷积层，输出通道为192、padding=1的3*3卷积层，stride=2、padding=1的3*3最大池化层
#     3、串联的2个inception块和一个MaxPooling层：
#         + 第一个inception块：4条路径的输出通道数分别为：64, (96, 128), (16, 32), 32
#         + 第二个inception块：4条路径的输出通道数分别为：128, (128, 192), (32, 96), 64
#         + stride=2、padding=1的3*3最大池化层
#     4、串联的5个inception块和一个MaxPooling层：
#         + 第一个inception块：4条路径的输出通道数为：192, (96, 208), (16, 48), 64
#         + 第二个inception块：4条路径的输出通道数为：160, (112, 224), (24, 64), 64
#         + 第三个inception块：4条路径的输出通道数为：128, (128, 256), (24, 64), 64
#         + 第四个inception块：4条路径的输出通道数为：112, (144, 288), (32, 64), 64
#         + 第五个inception块：4条路径的输出通道数为：256, (160, 320), (32, 128), 128
#         + stride=2、padding=1的3*3最大池化层
#     5、串联的2个inception块和一个AvgPooling层：
#         + 第一个inception块：4条路径的输出通道数为：256, (160, 320), (32, 128), 128
#         + 第二个inception块：4条路径的输出通道数为：384, (192, 384), (48, 128), 128
#         + 一个AvgPooling层，使输出的特征维度是1*1
#     6、一个线性层，使输出的维度是10
# lr=0.1
# optimizer=SGD
# test acc=85.36%

import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim


torch.manual_seed(23333)
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


class InceptionBlock(nn.Module):
    def __init__(self, in_channel, p1, p2, p3, p4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, p1, kernel_size=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, p2[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(p2[0], p2[1], kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channel, p3[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(p3[0], p3[1], kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(in_channel, p4, kernel_size=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        return torch.cat([conv1, conv2, conv3, conv4], dim=1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  # [B, 1, 96, 96]->[B, 64, 48, 48]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [B, 64, 48, 48]->[B, 64, 24, 24]
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),  # [B, 64, 24, 24]->[B, 192, 24, 24]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [B, 192, 24, 24]->[B, 192, 12, 12]
            InceptionBlock(192, 64, (96, 128), (16, 32), 32),  # [B, 192, 12, 12]->[B, 256, 12, 12]
            InceptionBlock(256, 128, (128, 192), (32, 96), 64),  # [B, 256, 12, 12]->[B, 480, 12, 12]
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [B, 480, 12, 12]->[B, 480, 6, 6]
            InceptionBlock(480, 192, (96, 208), (16, 48), 64),  # [B, 480, 6, 6]->[B, 512, 6, 6]
            InceptionBlock(512, 160, (112, 224), (24, 64), 64),  # [B, 512, 6, 6]->[B, 512, 6, 6]
            InceptionBlock(512, 128, (128, 256), (24, 64), 64),  # [B, 512, 6, 6]->[B, 512, 6, 6]
            InceptionBlock(512, 112, (144, 288), (32, 64), 64),  # [B, 512, 6, 6]->[B, 528, 6, 6]
            InceptionBlock(528, 256, (160, 320), (32, 128), 128),  # [B, 528, 6, 6]->[B, 832, 6, 6]
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [B, 832, 6, 6]->[B, 832, 3, 3]
            InceptionBlock(832, 256, (160, 320), (32, 128), 128),  # [B, 832, 3, 3]->[B, 832, 3, 3]
            InceptionBlock(832, 384, (192, 384), (48, 128), 128),  # [B, 832, 3, 3]->[B, 1024, 3, 3]
            nn.AvgPool2d(kernel_size=3)
        )
        self.lin = nn.Linear(1024, 10)

    def forward(self, x):
        conv = self.conv(x)
        return self.lin(conv.reshape(-1, 1024))
    

def init_weight(x):
    if type(x) == nn.Linear or type(x) == nn.Conv2d:
        nn.init.xavier_uniform_(x.weight)


net = GoogLeNet()
net.apply(init_weight)
net = net.to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)


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
    print(f'epoch {epoch + 1}, loss {epoch_loss:.6f}, acc {evaluate(net, test_data) * 100:.4f}%')
    
    
