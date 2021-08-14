from abc import ABC

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from LoadDataset import LoadBBDataset
from torch import optim
import math
import numpy as np


class DownConv(nn.Module, ABC):
    def __init__(self, chs_in, chs_out):
        super(DownConv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(chs_in),
            nn.Conv2d(chs_in, chs_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module, ABC):
    def __init__(self, chs_in, chs_out, output_padding):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.BatchNorm2d(chs_in),
            nn.ConvTranspose2d(chs_in, chs_out, kernel_size=4, padding=1, stride=2, bias=True,
                               output_padding=output_padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class DownConv3D(nn.Module, ABC):
    def __init__(self, in_chs, out_chs):
        super(DownConv3D, self).__init__()
        self.down3D = nn.Sequential(
            # nn.BatchNorm3d(in_chs),
            nn.Conv3d(in_channels=in_chs, out_channels=out_chs, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.down3D(x)
        return x


class UpConv3D(nn.Module, ABC):
    def __init__(self, in_chs, out_chs, output_padding):
        super(UpConv3D, self).__init__()
        self.up3D = nn.Sequential(
            # nn.BatchNorm3d(in_chs),
            nn.ConvTranspose3d(in_channels=in_chs, out_channels=out_chs, kernel_size=4, padding=1, stride=2, bias=True,
                               output_padding=output_padding),
            nn.ReLU(inplace=True)
        )
        self.Conv3D = DownConv3D(in_chs=in_chs, out_chs=out_chs)

    def forward(self, x0, x1):
        x0 = self.up3D(x0)
        x = torch.cat((x0, x1), dim=1)
        x = self.Conv3D(x)
        return x


class U_Net_2D(nn.Module, ABC):
    def __init__(self, img_chs, out_chs):
        super(U_Net_2D, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv0 = DownConv(chs_in=img_chs, chs_out=128)
        self.Conv1 = DownConv(chs_in=img_chs, chs_out=256)
        self.Conv2 = DownConv(chs_in=256, chs_out=512)
        self.Conv3 = DownConv(chs_in=512, chs_out=1024)

        self.Up3 = UpConv(chs_in=1024, chs_out=512, output_padding=0)
        self.Up_conv3 = DownConv(chs_in=1024, chs_out=512)

        self.Up2 = UpConv(chs_in=512, chs_out=256, output_padding=1)
        self.Up_conv2 = DownConv(chs_in=512, chs_out=256)

        self.Up1 = DownConv(chs_in=256, chs_out=128)
        self.Up_conv1 = DownConv(chs_in=256, chs_out=127)

        self.Conv_final = nn.Conv2d(in_channels=128, out_channels=out_chs, kernel_size=1, stride=1, bias=True,
                                    padding=0)
        self.Conv_zero = DownConv(chs_in=1, chs_out=1)

    def forward(self, x):
        zero_deg = torch.unsqueeze(x[:, 0, ...], dim=1)
        zero_deg = self.Conv_zero(zero_deg)
        factor = x[:, 1:, ...]
        x0 = self.Conv0(factor)
        x1 = self.Conv1(factor)
        # print(x1.shape)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        # print(x2.shape)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        # print(x3.shape)

        d3 = self.Up3(x3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        # print(d3.shape)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        # print(d2.shape)

        d1 = self.Up1(d2)
        d1 = torch.cat((x0, d1), dim=1)
        d1 = self.Up_conv1(d1)
        # print(d1.shape)

        d = self.Conv_final(torch.cat((zero_deg, d1), dim=1))
        d = nn.functional.softmax(d, dim=1)

        return d


class U_Net_3D(nn.Module, ABC):
    def __init__(self, in_chs, out_chs):
        super(U_Net_3D, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv0 = nn.Sequential(
            # nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
        self.Conv1 = DownConv3D(in_chs=1, out_chs=16)
        self.Conv2 = DownConv3D(in_chs=16, out_chs=32)
        self.Conv3 = DownConv3D(in_chs=32, out_chs=64)
        self.Conv4 = DownConv3D(in_chs=64, out_chs=128)
        self.Up3 = UpConv3D(in_chs=128, out_chs=64, output_padding=(0, 1, 0))
        self.Up2 = UpConv3D(in_chs=64, out_chs=32, output_padding=(0, 0, 0))
        self.Up1 = UpConv3D(in_chs=32, out_chs=16, output_padding=(0, 1, 1))
        self.Conv5 = DownConv3D(in_chs=16, out_chs=1)
        self.Conv_1x1 = nn.Conv2d(in_channels=in_chs, out_channels=out_chs, kernel_size=1, stride=1, padding=0,
                                  bias=True)

    def forward(self, zFactor, zeroDeg):
        zFactor = torch.unsqueeze(zFactor, dim=1)  # (batch_size, 1, 176, 101, 49)
        zeroDeg = torch.unsqueeze(zeroDeg, dim=1)
        zeroDeg = self.Conv0(zeroDeg)  # (batch_size, 1, 101, 49)

        x1 = self.Conv1(zFactor)  # (batch_size, 64, 176, 101, 49)

        x2 = self.Maxpool(x1)  # (batch_size, 64, 88, 50, 24)
        x2 = self.Conv2(x2)  # (batch_size, 128, 88, 50, 24)

        x3 = self.Maxpool(x2)  # (batch_size, 128, 44, 25, 12)
        x3 = self.Conv3(x3)  # (batch_size, 256, 44, 25, 12)

        f = self.Maxpool(x3)  # (batch_size, 256, 22, 12, 6)
        f = self.Conv4(f)  # (batch_size, 512, 22, 12, 6)

        f = self.Up3(f, x3)  # (batch_size, 256, 44, 25, 12)
        f = self.Up2(f, x2)  # (batch_size, 128, 88, 50, 24)
        f = self.Up1(f, x1)  # (batch_size, 64, 176, 101, 49)
        f = self.Conv5(f)  # (batch_size, 1, 176, 101, 49)

        f = torch.squeeze(f, dim=1)  # (batch_size, 176, 101, 49)
        f = self.Conv_1x1(torch.cat((zeroDeg, f), dim=1))  # (batch_size, 178, 101, 49)
        f = nn.functional.softmax(f, dim=1)

        return f


slice_width = 101
slice_num = 150
epoch = 20
batch_size = 12
lr = 0.02
loss_sum = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

GPM_BB_train_data = LoadBBDataset('data/train', slice_width, slice_num)
# GPM_BB_val_data = LoadBBDataset('data/val', slice_width, slice_num)
train_loader = DataLoader(GPM_BB_train_data, batch_size=batch_size, shuffle=False, num_workers=0)
# val_loader = DataLoader(GPM_BB_val_data, batch_size=batch_size, shuffle=False, num_workers=0)

# model = U_Net_3D(177, 3).to(device)
model = torch.load('model-epoch10-batch4000.pth').to(device)
# model = U_Net_3D(177, 1).to(device)
opt = optim.SGD(model.parameters(), lr=lr)
# Loss = nn.CrossEntropyLoss()
Loss = nn.CrossEntropyLoss(weight=torch.tensor([1, 20, 100], dtype=torch.float32).to(device))
# Loss = nn.MSELoss(reduction='mean')

# print(model)

for i in range(0, epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.float()
        output = model(data[:, 1:, ...], data[:, 0, ...])

        opt.zero_grad()
        loss = Loss(output, target.long())
        # loss = Loss(output, target.long()) * math.log((target.sum() + 10), 10)
        # loss = torch.log(Loss(output.float(), torch.unsqueeze(target, dim=1).float()) + 1)
        loss.backward()

        opt.step()

        loss_sum += loss
        loss_sum += math.exp(loss)
        times = 100
        if batch_idx % times == 0:
            Conv1_weight = model.Conv1.down3D[0].weight
            Conv_1x1_weight = model.Conv_1x1.weight
            print('Conv1 weight:\t\tMax:\t{}, Min:\t{}'.format(Conv1_weight.data.max(), Conv1_weight.data.min()))
            print('Conv_1x1 weight:\tMax:\t{}, Min:\t{}'.format(Conv_1x1_weight.data.max(), Conv_1x1_weight.data.min()))
            print('Conv1 grad:\t\t\tMax:\t{}, Min:\t{}'.format(Conv1_weight.grad.data.max(), Conv1_weight.grad.data.min()))
            print('Conv_1x1 grad:\t\tMax:\t{}, Min:\t{}'.format(Conv_1x1_weight.grad.data.max(),
                                                                Conv_1x1_weight.grad.data.min()))
            # print('epoch:{}, batch:{}, loss:{}, pure_loss:{}'.format(i + 1, batch_idx, loss,
            #                                                          loss / math.log((target.sum() + 10), 10)))
            print('epoch:{}, batch:{}, loss:{}'.format(i + 1, batch_idx, loss_sum/times))
            loss_sum = 0

        # if batch_idx % 50 == 0:
        #     print(output)
        if batch_idx % 2000 == 0:
            torch.save(model, 'model-epoch{}-batch{}.pth'.format(i + 1, batch_idx))
