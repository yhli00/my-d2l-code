# vgg11的pytorch实现
# vgg11由5个卷积块、2个线性层、1个输出层实现（输入图片的大小为224*224）：
#     + 5个卷积块的结构为：（conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))），
#       元组里的第一个数表示卷层数，第二个表示输出通道数
#         - 每个卷积块中使用3*3的卷积核，padding为1，使用ReLU激活函数
#         - 每个卷积块最后一层是核大小为2*2，步幅为2的最大池化层
#         - 在实现中为了加速训练，将原vgg中的通道数除以4
#     + 2个线性层和最后的输出层与AlexNet一致：
#         - 隐藏层大小均为4096，使用ReLU激活函数，使用丢弃概率为0.5的Dropout层
# num_epochs = 10
# lr = 0.05
# optimizer = SGD
# 使用xavier均匀分布初始化参数
# test acc=89.71%
# 疑问：去掉了net.apply(init_weight)后，模型训练时的test acc一直是10%？

import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader


device = torch.device('cuda')
torch.manual_seed(9999)

train_data = torchvision.datasets.FashionMNIST(
    root='../data',
    download=True,
    train=True,
    transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
)
test_data = torchvision.datasets.FashionMNIST(
    root='../data',
    download=True,
    train=False,
    transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
)
train_data = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=4)
test_data = DataLoader(test_data, batch_size=256, shuffle=True, num_workers=4)


class vgg11(nn.Module):
    def __init__(self, conv_arch):
        super().__init__()
        self.conv = self._vggConv(conv_arch)
        # 每经过一次卷积块，图片的高度和宽度变成原来的一般
        self.conv_arch = conv_arch
        self.lin = nn.Sequential(
            nn.Linear(conv_arch[-1][1] * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        conv = self.conv(x)
        return self.lin(conv.reshape(-1, self.conv_arch[-1][1] * 7 * 7))

    def _vggBlock(self, num_conv, in_channel, out_channel):
        layer = []
        in_ch, out_ch = in_channel, out_channel
        for _ in range(num_conv):
            layer.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layer.append(nn.ReLU())
            in_ch = out_ch
        layer.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layer)
    
    def _vggConv(self, conv_arch):
        in_ch = 1
        conv_layers = []
        for x, y in conv_arch:
            conv_layers.append(self._vggBlock(x, in_channel=in_ch, out_channel=y))
            in_ch = y
        return nn.Sequential(*conv_layers)

def init_weight(x):
    if type(x) == nn.Linear or type(x) == nn.Conv2d:
        nn.init.xavier_uniform_(x.weight)


conv_arch = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]
mini_conv_arch = [(x, y // 4) for x, y in conv_arch]
net = vgg11(mini_conv_arch)
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
