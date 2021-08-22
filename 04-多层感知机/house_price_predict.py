# 使用线性模型预测房价
# 使用5折交叉验证
# num_epochs=100
# lr=5
# batch_size=64
# weight_decay=0(不使用正规化)
# 使用平方对数均方差损失函数
# 注意：模型训练时用的是mse作为损失函数，评估时使用根号对数mse
# kaggle分数: 0.14839

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


torch.manual_seed(78910)

train_data_filename = '../data/house_price_predict/train.csv'
test_data_filename = '../data/house_price_predict/test.csv'

train_data = pd.read_csv(train_data_filename)
test_data = pd.read_csv(test_data_filename)
print(train_data.shape)
print(test_data.shape)
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]), axis=0)
print(all_features.shape)

num_index = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[num_index] = all_features[num_index].apply(lambda x: (x - x.mean()) / x.std())  # 对数字数据归一化
all_features[num_index] = all_features[num_index].fillna(0)
all_features = pd.get_dummies(all_features, dummy_na=True)  # 将离散数据转化成one-hot形式
print(all_features.shape[1])

loss_fn = nn.MSELoss()  # 模型的损失函数是mse
def sqrt_log_mse(y_hat, y):  # 模型的评估函数是sqrt_log_mse
    y_hat = torch.clamp(y_hat, min=1)
    return torch.sqrt(loss_fn(torch.log(y_hat), torch.log(y)))


def get_k_fold(k, i, x, y):
    nums = len(x) // k
    x_train, y_train = None, None
    x_valid, y_valid = None, None
    for j in range(k):
        index = slice(j * nums, (j + 1) * nums)
        if j == i:
            x_valid, y_valid = x[index], y[index]
        elif x_train is None:
            x_train, y_train = x[index], y[index]
        else:
            x_train = pd.concat((x_train, x[index]), axis=0)
            y_train = pd.concat((y_train, y[index]), axis=0)
    return x_train, y_train, x_valid, y_valid


class data_set(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = torch.tensor(x.values, dtype=torch.float32)
        self.y = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)


num_input = all_features.shape[1]
net = nn.Sequential(nn.Linear(num_input, 1))
optimizer = torch.optim.Adam(net.parameters(), weight_decay=0, lr=5)

train_features = all_features.iloc[:train_data.shape[0], :]
test_features = all_features.iloc[train_data.shape[0]:, :]
train_labels = train_data.iloc[:, -1]

def train_per_epoch(net, optimizer, x, y, k):
    net.train()
    total_train_loss = []
    total_valid_loss = []
    for i in range(k):
        x_trian, y_train, x_valid, y_valid = get_k_fold(k, i, x, y)
        train_data = DataLoader(data_set(x_trian, y_train), shuffle=True, num_workers=4, batch_size=64)
        for features, labels in train_data:
            y_hat = net(features)
            loss = loss_fn(y_hat, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            x_train_tensor = torch.tensor(x_trian.values, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
            x_valid_tensor = torch.tensor(x_valid.values, dtype=torch.float32)
            y_valid_tensor = torch.tensor(y_valid.values.reshape(-1, 1), dtype=torch.float32)
            train_loss = sqrt_log_mse(net(x_train_tensor), y_train_tensor)
            valid_loss = sqrt_log_mse(net(x_valid_tensor), y_valid_tensor)
            total_train_loss.append(train_loss.item())
            total_valid_loss.append(valid_loss.item())
    return total_train_loss, total_valid_loss
            

def train(net, optimizer, num_epochs, x, y, k):
    for epoch in range(num_epochs):
        total_train_loss, total_valid_loss = train_per_epoch(net, optimizer, x, y, k)
        for idx, (i, j) in enumerate(zip(total_train_loss, total_valid_loss)):
            print(f'epoch {epoch + 1}, fold {idx + 1}, train_loss {i:.6f}, valid_loss {j:.6f}')
        print(f'epoch {epoch + 1}, avg_train_loss {sum(total_train_loss) / k}, avg_valid_loss {sum(total_valid_loss) / k}')

def predict(net, test_features):
    net.eval()
    test_tensor = torch.tensor(test_features.values, dtype=torch.float32)
    output = net(test_tensor)
    output = output.squeeze()
    submission = pd.DataFrame()
    submission['Id'] = test_data['Id']
    submission['SalePrice'] = pd.Series(output.detach().clone().numpy())
    submission.to_csv('../data/house_price_predict/submission.csv', index=False)


train(net, optimizer, 100, train_features, train_labels, 5)
predict(net, test_features)