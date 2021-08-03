from abc import ABC

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from LoadDataset import LoadBBDataset
import matplotlib.pyplot as plt


class DownConv3D(nn.Module, ABC):
    def __init__(self, in_chs, out_chs):
        super(DownConv3D, self).__init__()
        self.down3D = nn.Sequential(
            nn.BatchNorm3d(in_chs),
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
            nn.BatchNorm3d(in_chs),
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

class U_Net_3D(nn.Module, ABC):
    def __init__(self, in_chs, out_chs):
        super(U_Net_3D, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv0 = nn.Sequential(
            nn.BatchNorm2d(1),
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
        self.Conv_1x1 = nn.Conv2d(in_channels=in_chs, out_channels=out_chs, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, zFactor, zeroDeg):
        zFactor = torch.unsqueeze(zFactor, dim=1)   # (batch_size, 1, 176, 101, 49)
        zeroDeg = torch.unsqueeze(zeroDeg, dim=1)
        zeroDeg = self.Conv0(zeroDeg)         # (batch_size, 1, 101, 49)

        x1 = self.Conv1(zFactor)    # (batch_size, 64, 176, 101, 49)

        x2 = self.Maxpool(x1)       # (batch_size, 64, 88, 50, 24)
        x2 = self.Conv2(x2)         # (batch_size, 128, 88, 50, 24)

        x3 = self.Maxpool(x2)       # (batch_size, 128, 44, 25, 12)
        x3 = self.Conv3(x3)         # (batch_size, 256, 44, 25, 12)

        f = self.Maxpool(x3)        # (batch_size, 256, 22, 12, 6)
        f = self.Conv4(f)           # (batch_size, 512, 22, 12, 6)

        f = self.Up3(f, x3)         # (batch_size, 256, 44, 25, 12)
        f = self.Up2(f, x2)         # (batch_size, 128, 88, 50, 24)
        f = self.Up1(f, x1)         # (batch_size, 64, 176, 101, 49)
        f = self.Conv5(f)           # (batch_size, 1, 176, 101, 49)

        f = torch.squeeze(f, dim=1)     # (batch_size, 176, 101, 49)
        f = self.Conv_1x1(torch.cat((zeroDeg, f), dim=1))   # (batch_size, 178, 101, 49)
        f = nn.functional.softmax(f, dim=1)

        return f


slice_width = 101
slice_num = 150
epoch = 10
batch_size = 1

cols, rows = 2, 6
figure = plt.figure(figsize=(20, rows*5))
i = 0
GPM_BB_data = LoadBBDataset('data/train', slice_width, slice_num)
data_loader = DataLoader(GPM_BB_data, batch_size=batch_size, shuffle=True, num_workers=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('model-epoch3-batch2000.pth').to(device)
for batch_idx, (data, target) in enumerate(data_loader):
    data = data.float().to(device)
    target = target.float()
    with torch.no_grad():
        output = model(data[:, 1:, ...], data[:, 0, ...])

        # print(torch.squeeze(output.cpu(), dim=0))
        output = torch.argmax(torch.squeeze(output.cpu(), dim=0), 0)
        # output = output.cpu().view(101, 49)
        print(output)

    if target.sum() > 1000:
        i += 1
        if 1 <= i <= rows:
            figure.add_subplot(rows, cols, i*2-1)
            # plt.axis("off")
            plt.imshow(torch.squeeze(target, dim=0).T, cmap="gray")
            figure.add_subplot(rows, cols, i*2)
            # plt.axis("off")
            plt.imshow(output.T, cmap="gray")

    if i >= rows:
        break
plt.show()