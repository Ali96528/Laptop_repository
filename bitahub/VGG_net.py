from tensorboardX import SummaryWriter
import os
from datetime import datetime
import torch.nn as nn
from torch.utils import data
import torchvision
import torch


def vgg_block(num_of_conv, in_channels, out_channnels):
    """vgg块: (1个3x3卷积层,1个relu层)xn+一个maxpool层2x2"""
    layers = []
    for _ in range(num_of_conv):
        layers.append(nn.Conv2d(in_channels, out_channnels,
                      kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channnels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg(conv_arch):
    """
        VGG-11:8个卷积层和3个全连接层
        VGG网络,conv_arch定义每个VGG块中卷积层和输出通道数量
        有五个卷积块,前两个块各一个卷积层,后三个块各两个卷积层
        第一个块64个输出通道,逐个翻倍到达512
    """
    conv_blocks = []
    in_channels = 1  # 这里使用的是f-mnist，所以是单层的灰度图像输入
    # 卷积层部分
    for (num_of_convs, out_channels) in conv_arch:
        conv_blocks.append(vgg_block(num_of_convs, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(*conv_blocks,
                         nn.Flatten(),
                         nn.Linear(in_channels*7*7, 4096),
                         nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.Linear(4096, 4096),
                         nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.Linear(4096, 10)
                         )


# 指定卷积层个数和输出通道数，一个conv_arch元素代表一个块，这里使用VGG-11网络
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

"""
    # 搭建网络
    net = vgg(conv_arch)
    # 构建单通道数据样本，以观察每个层的输出的形状
    X = torch.randn(size=(1, 1, 224, 224))
    # 打印输出查看形状
    for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, 'output shape:\t', X.shape)
"""


# 正常VGG运算量太大，缩小模型减少运算量
ratio = 4  # 缩放比例
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)


# 显示运算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Devices:', device)

# 超参数设置
epoch = 10
Batch_Size = 256
learning_rate = 0.01
img_resize = 224
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
    test_img = torch.rand(5, 1, 224, 224)
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
