# AlexNet的pytorch实现
# AlexNet由3个部分组成（5层的卷积层、两层的全连接层、一个输出层，输入图像的大小为224*224，使用ReLu激活函数）：
#     + 第一个卷积层：96个输出通道、卷积核大小11、步幅为4、padding为1。后接核大小为3步幅为2的max池化层
#     + 第二个卷积层：256个输出通道、卷积核大小为5、步幅为1、padding为2。后接核大小为3步幅为2的max池化层
#     + 第三个卷积层：384个输出通道、卷积核大小为3、步幅为1、padding为1
#     + 第四个卷积层：384个输出通道、卷积核大小为3、步幅为1、padding为1
#     + 第五个卷积层：256个输出通道、卷积核大小为3、步幅为1、padding为1。后接和大小为3步幅为2的max池化层
#     + 第一个全连接层：输出大小为4096。后接丢弃概率为0.5的dropout层
#     + 第二个全连接层：输出大小为4096。后接丢弃概率为0.5的dropout层
#     + 输出层
# num_epochs = 10
# batch_size = 128
# lr = 0.01
# optimizer = SGD
# 使用xavier均匀分布初始化模型参数
# test acc=0.859

import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms


torch.manual_seed(8888)
device = torch.device('cuda')

train_data = torchvision.datasets.FashionMNIST(root='../data', download=True, train=True, 
                                               transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
test_data = torchvision.datasets.FashionMNIST(root='../data', download=True, train=False, 
                                              transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
train_data = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=4)
test_data = DataLoader(test_data, shuffle=True, batch_size=256, num_workers=4)

print(next(iter(train_data))[0].shape)
print(next(iter(train_data))[1].shape)

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, padding=1, stride=4),  # [B, 1, 224, 224]->[B, 96, 54, 54]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # [B, 96, 54, 54]->[B, 96, 26, 26]
            nn.Conv2d(96, 256, kernel_size=5, padding=2),  # [B, 96, 26, 26]->[B, 256, 26, 26]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # [B, 256, 26, 26]->[B, 256, 12, 12]
            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # [B, 256, 12, 12]->[B, 384, 12, 12]
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # [B, 384, 12, 12]->[B, 384, 12, 12]
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # [B, 384, 12, 12]->[B, 384, 12, 12]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)  # [B, 384, 12, 12]->[B, 256, 5, 5]
        )
        self.lin = nn.Sequential(
            nn.Linear(6400, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10)
        )
    
    def forward(self, x):
        conv = self.conv(x)
        return self.lin(conv.reshape(-1, 6400))


def init_weight(x):
    if type(x) == nn.Linear or type(x) == nn.Conv2d:
        nn.init.xavier_uniform_(x.weight)


net = AlexNet()
net.apply(init_weight)
net = net.to(device)


optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()


def evaluate(net, test_data):
    net.eval()
    total_num = 0.
    correct = 0.
    for x, y in test_data:
        x = x.to(device)
        y = y.to(device)
        y_hat = net(x)
        y_hat = torch.argmax(y_hat, dim=1)
        correct += torch.sum(y_hat.type(y.dtype) == y).float().item()
        total_num += len(x)
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
