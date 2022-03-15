
from tensorboardX import SummaryWriter
import os
from datetime import datetime
import torch.nn as nn
from torch.utils import data
import torchvision
import torch
from torch.nn import functional as F


class Inception_block(nn.Module):
    """
        前三条路径使用窗口大小为 1x1 、 3x3 和 5x5 的卷积层，从不同空间大小中提取信息**。 
        中间的两条路径在输入上执行 1x1 卷积，以减少通道数，从而降低模型的复杂性。
        第四条路径使用 3x3 最大汇聚层，然后使用 1x1 卷积层来改变通道数 
        C1-C4是每条路径的输出通道数
    """

    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception_block, self).__init__(**kwargs)
        # 线路1,单个1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2,1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3,1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4,3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))  # 通过池化，所以不用relu
        # 在通道上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)


b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(), nn.Conv2d(
    64, 192, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b3 = nn.Sequential(Inception_block(192, 64, (96, 128), (16, 32), 32),
                   Inception_block(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b4 = nn.Sequential(Inception_block(480, 192, (96, 208), (16, 48), 64),
                   Inception_block(512, 160, (112, 224), (24, 64), 64),
                   Inception_block(512, 128, (128, 256), (24, 64), 64),
                   Inception_block(512, 112, (144, 288), (32, 64), 64),
                   Inception_block(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b5 = nn.Sequential(Inception_block(832, 256, (160, 320), (32, 128), 128),
                   Inception_block(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1, 1)),
                   nn.Flatten())

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))


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
