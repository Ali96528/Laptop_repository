
import torch
import torch.optim
import torchvision
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils import data

writer = SummaryWriter('C:/Users/libangmao/Desktop/Lenet')


def get_dataloader_workers():
    """加载数据集时使用的进程数"""
    return 8


def load_data(batch_size, resize=None):
    """获得训练集数据,并迭代返回"""
    if resize:

        mnist_train_data = torchvision.datasets.FashionMNIST(
            root='C:/Users/libangmao/Desktop/test_dataset',
            train=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(resize),
                torchvision.transforms.ToTensor()
            ]),
            download=True)
        mnist_test_data = torchvision.datasets.FashionMNIST(
            root='C:/Users/libangmao/Desktop/test_dataset',
            train=False,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(resize),
                torchvision.transforms.ToTensor()
            ]),
            download=True)
    else:
        mnist_train_data = torchvision.datasets.FashionMNIST(
            root='C:/Users/libangmao/Desktop/test_dataset',
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True)
        mnist_test_data = torchvision.datasets.FashionMNIST(
            root='C:/Users/libangmao/Desktop/test_dataset',
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True)
    return data.DataLoader(mnist_train_data, batch_size=batch_size, shuffle=True, num_workers=get_dataloader_workers()), data.DataLoader(mnist_test_data, batch_size=batch_size, shuffle=True, num_workers=get_dataloader_workers())


class Accumulator():
    """计数器"""

    def __init__(self, n):
        self.data = [0.0]*n

    def add(self, *args):
        self.data = [a+float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0]*len(self.data)

    def __getitem__(self, id):
        return self.data[id]


def accuracy(y_hat, y):
    # calculate how many samples are predicted correctlly.
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    # cmp is a sign to compare the type of y_hat and y
    #  make sure y_hat has the same type with y
    return float(cmp.type(y.dtype).sum())
    # if these two types are different ,then we get 0


def evaluate_accuracy(net_model, data_iter):
    """计算模型精度"""
    if isinstance(net_model, nn.Module):
        net_model.eval()  # 设置为评估模式
    metric = Accumulator(2)  # 正确数，样本总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net_model(X), y), y.numel())
    return metric[0]/metric[1]


def train(net_model, train_iter, test_iter, num_of_epochs, lr):
    """训练函数"""
    # 如果是线性层或者卷积层就使用Xavier作为初始权重设置。
    if type(net_model) == nn.Linear or type(net_model) == nn.Conv2d:
        nn.init.xavier_uniform_(net_model.weight)
    optimizer = torch.optim.SGD(net_model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    count = 0
    if __name__ == '__main__':
        for epoch in range(num_of_epochs):
            metric = Accumulator(3)  # 损失和，准确率和，样本数
            net_model.train()  # 设置为训练模式
            for X, y in train_iter:
                optimizer.zero_grad()
                y_hat = net_model(X)
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()
                with torch.no_grad():
                    metric.add(l*X.shape[0], accuracy(y_hat, y), X.shape[0])
                train_loss = metric[0]/metric[2]
                train_acc = metric[1]/metric[2]
                count = count+1
                writer.add_scalars('figure1',
                                   {'train_loss': train_loss, 'train_acc': train_acc}, count)
            test_acc = evaluate_accuracy(net_model, test_iter)
            writer.add_scalar('test_accuracy', test_acc, count)
        print(f'loss {train_loss:.3f}, train acc {train_acc:.3f}, '
              f'test acc {test_acc:.3f}')


net_model = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(6400, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 10)
)  # 用的是fashion-mnist 所以最后类目是10而不是1000


num_of_epochs = 10
lr = 0.01
batch_size = 128
train_iter, test_iter = load_data(batch_size=batch_size, resize=224)
train(net_model, train_iter, test_iter, num_of_epochs, lr)
