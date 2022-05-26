
import torch
from torch.utils import data  # 获取迭代数据
from torch.autograd import Variable  # 获取变量
import torchvision
# from torchvision.datasets import mnist # 获取数据集
import matplotlib.pyplot as plt
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import torchvision.utils as vutils

import torch.nn as nn

# 输入模型的图片大小
imageSize = 256
batchSize = 16
# 分类数
numclasses = 4
# 训练数据集
traindataset = dset.ImageFolder(root="/Users/touristk/Apple_data/train",
                                transform=transforms.Compose([
                                    transforms.Resize(imageSize),
                                    transforms.CenterCrop(imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
# 训练数据集dataloader
traindataloader = torch.utils.data.DataLoader(traindataset,
                                              batch_size=batchSize,
                                              shuffle=True,
                                              # num_workers=int(opt.workers)
                                              )

# 验证数据集
testdataset = dset.ImageFolder(root="/Users/touristk/Apple_data/test",
                               transform=transforms.Compose([
                                   transforms.Resize(imageSize),
                                   transforms.CenterCrop(imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

# 验证数据集dataloader
testdataloader = torch.utils.data.DataLoader(testdataset,
                                             batch_size=batchSize,
                                             shuffle=True,
                                             # num_workers=int(opt.workers)
                                             )


class inception(nn.Module):
    def __init__(self, num_channels):
        super(inception, self).__init__()

        strides = 1
        self.conv3 = nn.Sequential(
            # dconv(num_channels,num_channels,k=1,s=1,p=0),
            nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            # dconv(num_channels,num_channels,k=3,s=1,p=1),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            # dconv(num_channels,num_channels,k=5,s=1,p=2),
            nn.Conv2d(num_channels, num_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )

    def forward(self, input):
        X = self.conv1(input)
        Y = self.conv3(input)
        Z = self.conv5(input)
        C = torch.cat((X, Y), dim=1)
        D = torch.cat((C, Z), dim=1)
        return D


class resblock(nn.Module):  # 本类已保存在d2lzh包中方便以后使用
    def __init__(self, num_channels):
        super(resblock, self).__init__()

        strides = 1
        self.resblock = nn.Sequential(
            # dconv(num_channels,num_channels,k=3,s=1,p=1),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            # dconv(num_channels,num_channels,k=3,s=1,p=1),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )

    def forward(self, input):
        Y = self.resblock(input)
        X = input
        return X + Y


class block(nn.Module):
    def __init__(self, num_channels):
        super(block, self).__init__()

        strides = 1
        self.inception = inception(num_channels)
        self.resblock = resblock(num_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels * 4, num_channels, kernel_size=1, stride=1, padding=0),
            # dconv(num_channels*4,num_channels,k=1,s=1,p=0),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )

    def forward(self, input):
        X = self.inception(input)
        Y = self.resblock(input)
        output = torch.cat((X, Y), dim=1)

        output = self.conv(output)
        return output


# 定义网络结构
# O=(I-K+2P)/S+1;
class CNNnet(nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        # 256
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # resblock(16),
            block(16),
        )
        # 128
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # resblock(32),
            block(32),
        )
        # 64
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # resblock(64),
            block(64),
        )
        # 32
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # resblock(64),
            block(128),
        )
        # 16
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # resblock(64),
            block(256),
        )
        # 8
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # resblock(64),
            block(512),
        )
        # 4
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # resblock(64),
            block(512),
        )
        '''
        self.mlp1 = nn.Linear(2 * 2 * 64, 100)
        self.mlp2 = nn.Linear(100, numclasses)
        '''
        self.mlp1 = nn.Linear(2 * 2 * 512, 100)
        self.mlp2 = nn.Linear(100, numclasses)
        # self.mlp3=nn.Linear(100,numclasses)
        # self.softmax=nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        # 进全连接层之前需要把特征图拉伸
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.mlp2(x)
        x = self.sigmoid(x)
        # x = self.mlp3(x)
        # x = self.softmax(x)
        return x


# 实例化模型
model = CNNnet()
# 输出模型结构
print(model)
# 实例化交叉熵损失函数
loss_func = nn.CrossEntropyLoss()
# 使用优化器
opt = torch.optim.Adam(model.parameters(), lr=0.0001)
# opt= torch.optim.RMSprop(model.parameters(), lr=0.0001, alpha=0.9)


# 训练模型
epochs = 1

loss_count = []
for epoch in range(epochs):
    print("epoch：", epoch)
    # 遍历数据集
    for i, (x, y) in enumerate(traindataloader):
        # 图片
        batch_x = Variable(x)  # torch.Size([128, 1, 28, 28])
        # 标签
        batch_y = Variable(y)  # torch.Size([128])

        # 获取模型输出
        out = model(batch_x)  # torch.Size([128,10])
        # 获取损失
        loss = loss_func(out, batch_y)

        # 使用优化器优化
        opt.zero_grad()  # 清空上一步残余更新参数值
        loss.backward()  # 误差反向传播，计算参数更新值
        opt.step()  # 将参数更新值施加到net的parmeters上
        loss_count.append(loss)
        print("损失值：", loss.item())
        # torch.save(model)
        # print('Finished Training')
        if True:
            for a, b in testdataloader:
                # 图片
                test_x = Variable(a)
                # 标签
                test_y = Variable(b)
                # 获取当前数据的batch
                s, _, _, _ = test_x.shape

                print("实际标签", test_y)
                out = model(test_x)
                predict_value, predict_idx = torch.max(out, 1)
                # predict_value = torch.sigmoid(predict_value)

                # predict_value=torch.softmax(predict_value)
                print("预测准确率", predict_value)
                print("预测标签", predict_idx)

torch.save(model, '')
# model.load_state_dict(torch.load(''))

'''
                plt.figure(figsize=(8, 8))
                plt.axis("off")
                plt.title("Train image")
                plt.imshow(np.transpose(vutils.make_grid(test_x, padding=2, normalize=True).cpu(), (1, 2, 0)))
                plt.show()
'''

'''        
plt.figure('PyTorch_CNN_Loss')
plt.plot(loss_count, label='Loss')
plt.legend()
plt.show()
'''
