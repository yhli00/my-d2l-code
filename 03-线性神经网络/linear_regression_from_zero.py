# 训练数据集：w=[2, -3.4], b=4.2, 噪声由均值为0方差0.01的生态分布生成, 数据由均值为0方差为1的生态分布生成，共1000条数据
# epochs=3
# lr=0.03
# batch_size=10
# w由均值为0方差为0.01的正态分布初始化，b初始化为0

import random
import torch


def generate_data(w, b, nums):
    x = torch.normal(0, 1, (nums, len(w)))
    y = torch.matmul(x, w.reshape(len(w), -1)) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y

def data_iter(features, labels, batch_size):
    indics = list(range(len(labels)))
    random.shuffle(indics)
    for i in range(0, len(labels), batch_size):
        index = indics[i:min(i + batch_size, len(labels))]
        yield features[index], labels[index]

def mse_loss(y_hat, y):
    return sum((y_hat - y)**2) / (2 * len(y))

def sgd(params, lr):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad
            param.grad.zero_()

def linear_regression(x, w, b):
    return torch.matmul(x, w.reshape(len(w), -1)) + b


nums = 1000
true_w = torch.tensor([2, -3.4])
true_b = torch.tensor(4.2)
features, labels = generate_data(true_w, true_b, nums)
batch_size = 10
epochs = 3
lr = 0.03
w = torch.normal(0, 0.01, (2,), requires_grad=True)
b = torch.tensor(0., requires_grad=True)
net = linear_regression
for epoch in range(epochs):
    for x, y in data_iter(features, labels, batch_size):
        y_hat = net(x, w, b)
        loss = mse_loss(y_hat, y)
        loss.backward()
        sgd([w, b], lr)
    with torch.no_grad():
        total_loss = mse_loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {total_loss.item():.6f}')
print(w, b)
print(true_w, true_b)