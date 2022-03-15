
from tensorboardX import SummaryWriter
import os
from datetime import datetime
import torch.nn as nn
from torch.utils import data
import torchvision
import torch
from torch.nn import functional as F


class Residual(nn.Module):
    """
        3x3卷积+batchnorm+ReLu+3x3卷积+batchnorm
    """

    def __init__(self, in_channels, num_of_channels, use_1x1conv=False, strides=1):
        super().__init__()
        # 线路1,单个1x1卷积层
        self.conv1 = nn.Conv2d(in_channels, num_of_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(
            num_of_channels, num_of_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(
                in_channels, num_of_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_of_channels)
        self.bn2 = nn.BatchNorm2d(num_of_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(), nn.Linear(512, 10))


# 显示运算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Devices:', device)

# 超参数设置
epoch = 10
Batch_Size = 256
learning_rate = 0.1
img_resize = 96
num_load_workers = 8  # dataloader中的加载线程数量


# 加载时候的一系列操作
if img_resize is not None:
    transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_resize),
        torchvision.transforms.ToTensor()])
else:
    transformation = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])

# 加载数据集 使用torchvision自带函数
output_dir = 'C:/Users/libangmao/Desktop/output'
data_dir = 'C:/Users/libangmao/Desktop/test_dataset/FashionMNIST'
train_set = torchvision.datasets.FashionMNIST(
    data_dir, train=True, transform=transformation, target_transform=None, download=False)
test_set = torchvision.datasets.FashionMNIST(
    data_dir, train=False, transform=transformation, target_transform=None, download=False)
# 打印数据集和测试集长度：
print('train set num:', len(train_set), '\ntest set num:', len(test_set))
# 数据迭代器
train_iter = data.DataLoader(
    train_set, batch_size=Batch_Size, shuffle=True, num_workers=num_load_workers)
test_iter = data.DataLoader(
    test_set, batch_size=Batch_Size, shuffle=True, num_workers=num_load_workers)


# 损失函数 这里为交叉熵，多用于多分类问题
loss_function = nn.CrossEntropyLoss()
# 优化器 这里使用Adam L2范式权值缩减暂时不使用,学习率
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)


# 训练
if __name__ == '__main__':
    print('Ready to train!')
    # 使用样本先生成模型的参数维度，打印以供观察,这里使用五张图片
    test_img = torch.rand(1, 1, 96, 96)
    test_data = test_img
    for layer in net:
        test_data = layer(test_data)
        print(layer.__class__.__name__, 'output shape:\t', test_data.shape)
    # 日志输出目录
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join('/output', 'logs', current_time+'_AlexNet')
    # 模型参数初始化

    def init_weights(layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            nn.init.xavier_uniform_(layer.weight, gain=1)

    net.apply(init_weights)
    # 开始训练并记录
    with SummaryWriter(logdir) as writer:
        writer.add_graph(net, (test_img))  # 画出模型图
        print('ok')
        net.to(device)  # 模型转移到GPU中
        for epoch in range(epoch):
            print('\nEpoch: %d' % (epoch + 1))
            net.train()  # 开启训练模式
            sum_loss = 0.0  # 总损失计数器
            correct = 0.0  # 分类正确计数器
            total = 0.0  # 总样本数
            for i, data in enumerate(train_iter):
                # 数据准备工作
                length = len(train_iter)
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # 清空梯度
                optimizer.zero_grad()
                # forward + backward
                outputs = net(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                # 更新梯度
                optimizer.step()
                # 每训练1个batch打印一次loss和准确率
                sum_loss += loss.item()  # loss.item()是均值/标量，这里获取相加
                # 不保存每行的最大值，只保存每行最大值的索引
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum().float()
                # 记录损失以及训练精度
                print('[epoch:%d, iter:%d] curr_Loss: %0.3f Loss: %.03f | Acc: %.3f%% '
                      % (epoch + 1, (i + 1 + epoch * length), loss.item(), sum_loss / (i + 1), 100. * correct / total))
                writer.add_scalar('scalar/train_loss',
                                  sum_loss / (i + 1), (i + 1 + epoch * length))
                writer.add_scalar('scalar/train_acc', 100. *
                                  correct / total, (i + 1 + epoch * length))
            # 一个epoch训练结束，查看学习率并调整
            for param in optimizer.param_groups:
                output = []
                for k, v in param.items():
                    if k == 'lr':
                        print('%s:%s' % (k, v))
                        writer.add_scalar('scalar/learning_rate', v, (epoch+1))
            # 记录参数信息
            for name, param in net.named_parameters():
                if 'bn' not in name:  # 非batch_norm层时
                    writer.add_histogram(name, param, epoch+1)
            # writer.add_image('Image', x, n_iter)
            writer.add_text('Text', 'text logged at step:' +
                            str(epoch + 1), epoch + 1)
            # 每训练完一个epoch测试一下准确率
            print("Now Test!")
            with torch.no_grad():
                correct = 0
                total = 0
                for data in test_iter:
                    net.eval()
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    # 取得分最高的那个类 (outputs.data的索引号)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().float()

                print('test acc is %.3f%%' % (100 * correct / total))
                acc = 100. * correct / total
                writer.add_scalar('scalar/trainval_acc', acc, epoch + 1)
                # 将每次测试结果实时写入acc.txt文件中
        print('Saving model......')
        torch.save(net.state_dict(), '%s/final_net.pth' % (output_dir))
        print("Training Finished, TotalEPOCH=%d" % epoch)
