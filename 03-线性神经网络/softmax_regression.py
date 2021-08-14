import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

def evaluate(net, test_data):
    with torch.no_grad():
        correct = 0.
        total_nums = 0.
        for x, y in test_data:
            y_hat = net(x.reshape(-1, 784))
            labels_hat = y_hat.argmax(dim=1)
            labels_hat.type(y.dtype)
            correct += torch.sum(labels_hat == y).float()
            total_nums += len(y)
    return (correct / total_nums).item()


train_data = torchvision.datasets.FashionMNIST(root='../data', transform=transforms.ToTensor(), download=True, train=True)
test_data = torchvision.datasets.FashionMNIST(root='../data', transform=transforms.ToTensor(), download=True, train=False)
train_data = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=4)
test_data = DataLoader(test_data, batch_size=256, shuffle=True, num_workers=4)

net = nn.Sequential(nn.Linear(784, 10))
net[0].weight.data.normal_(0, 0.1)
net[0].bias.data.fill_(0)

loss = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.SGD(net.parameters(), lr=0.1)
num_epochs = 10


test_acc = evaluate(net, test_data)
print(f'test_acc {test_acc:.4f}')
for epoch in range(num_epochs):
    total_loss = torch.tensor([])
    for x, y in train_data:
        y_hat = net(x.reshape(-1, 784))
        lo = loss(y_hat, y)
        total_loss = torch.cat((total_loss, lo.detach().clone()), dim=0)
        optimizer.zero_grad()
        lo.mean().backward()
        optimizer.step()
    with torch.no_grad():
        test_acc = evaluate(net, test_data)
        avg_loss = torch.sum(total_loss) / len(total_loss)
        print(f'epoch {epoch + 1},  loss {avg_loss.item():.6f},  test_acc {test_acc:.4f}')