import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Vgg16_net(nn.Module):
    def __init__(self):
        super(Vgg16_net, self).__init__()

        # 第一层卷积层
        self.layer1 = nn.Sequential(
            # 输入3通道图像，输出64通道特征图，卷积核大小3x3，步长1，填充1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            # 对64通道特征图进行Batch Normalization
            nn.BatchNorm2d(64),
            # 对64通道特征图进行ReLU激活函数
            nn.ReLU(inplace=True),
            # 输入64通道特征图，输出64通道特征图，卷积核大小3x3，步长1，填充1
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            # 对64通道特征图进行Batch Normalization
            nn.BatchNorm2d(64),
            # 对64通道特征图进行ReLU激活函数
            nn.ReLU(inplace=True),

            # 进行2x2的最大池化操作，步长为2
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 第二层卷积层
        self.layer2 = nn.Sequential(
            # 输入64通道特征图，输出128通道特征图，卷积核大小3x3，步长1，填充1
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            # 对128通道特征图进行Batch Normalization
            nn.BatchNorm2d(128),
            # 对128通道特征图进行ReLU激活函数
            nn.ReLU(inplace=True),

            # 输入128通道特征图，输出128通道特征图，卷积核大小3x3，步长1，填充1
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            # 对128通道特征图进行Batch Normalization
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 进行2x2的最大池化操作，步长为2
            nn.MaxPool2d(2, 2)
        )
        # 第三层卷积层
        self.layer3 = nn.Sequential(
            # 输入为128通道，输出为256通道，卷积核大小为33，步长为1，填充大小为1
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 批归一化
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)
        )

        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        # 对张量的拉平(flatten)操作，即将卷积层输出的张量转化为二维，全连接的输入尺寸为512
        x = x.view(-1, 512)
        x = self.fc(x)
        return x