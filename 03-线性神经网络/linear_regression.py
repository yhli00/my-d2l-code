# 训练数据集：w=[2, -3.4], b=4.2, 噪声由均值为0方差0.01的生态分布生成, 数据由均值为0方差为1的生态分布生成，共1000条数据
# epochs=3
# lr=0.03
# batch_size=10
# w由均值为0方差为0.01的正态分布初始化，b初始化为0

import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader


def generate_data(w, b, nums):
    x = torch.normal(0, 1, (nums, len(w)))
    y = torch.matmul(x, w.reshape(len(w), -1)) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y


true_w = torch.tensor([2, -3.4])
true_b = torch.tensor(4.2)
features, labels = generate_data(true_w, true_b, 1000)
train_data = DataLoader(TensorDataset(features, labels), shuffle=True, batch_size=10)

net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.1)
net[0].bias.data.fill_(0)
loss = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.03)
epochs = 3

for epoch in range(epochs):
    for x, y in train_data:
        y_hat = net(x)
        lo = loss(y_hat, y)
        optimizer.zero_grad()
        lo.backward()
        optimizer.step()
    total_loss = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {total_loss.item():.6f}')
print(true_w, true_b)
print(net[0].weight.data, net[0].bias.data)
