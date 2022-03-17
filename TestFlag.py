import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from LoadDataset import LoadBBDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from U_Net import U_Net_3D


class ChannelAttention2D(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Linear(in_planes, in_planes // ratio),
                                nn.BatchNorm1d(in_planes // ratio),
                                nn.ReLU(),
                                nn.Linear(in_planes // ratio, in_planes),
                                nn.BatchNorm1d(in_planes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.avg_pool(x).view(-1, x.shape[1])
        avg_out = self.fc(avg)
        max = self.max_pool(x).view(-1, x.shape[1])
        max_out = self.fc(max)
        out = avg_out + max_out
        return self.sigmoid(out.unsqueeze(dim=-1).unsqueeze(dim=-1))


class SpatialAttention2D(nn.Module):
    def __init__(self, kernel_size=3, padding=1):
        super(SpatialAttention2D, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.BN = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.BN(x)
        return self.sigmoid(x)


class FactorSpatialAttention2D(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size=3, padding=1):
        super(FactorSpatialAttention2D, self).__init__()

        self.conv1 = nn.Conv2d(in_chs, 32, kernel_size, padding=padding, bias=True)
        self.BN = nn.BatchNorm2d(32)
        self.tanh = nn.Tanh()
        self.conv2 = nn.Conv2d(32, out_chs, kernel_size, padding=padding, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.BN(x)
        x = self.tanh(x)
        x = self.conv2(x)
        return self.sigmoid(x)


class DownConv3D(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(DownConv3D, self).__init__()
        self.down3D = nn.Sequential(
            nn.Conv3d(in_channels=in_chs, out_channels=in_chs, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(in_chs),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=in_chs, out_channels=out_chs, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_chs),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_chs, out_channels=out_chs, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(out_chs),
            nn.ReLU(inplace=True),
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv3d(in_channels=in_chs, out_channels=out_chs, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(out_chs),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        shortcut = self.conv1x1(x)
        x = self.down3D(x)
        return shortcut + x


class UpConv3D(nn.Module):
    def __init__(self, in_chs, out_chs, output_padding):
        super(UpConv3D, self).__init__()
        self.up3D = nn.Sequential(
            nn.Conv3d(in_channels=in_chs, out_channels=in_chs, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(in_chs),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(in_channels=in_chs, out_channels=out_chs, kernel_size=4, padding=1, stride=2, bias=True,
                               output_padding=output_padding),
            nn.BatchNorm3d(out_chs),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_chs, out_channels=out_chs, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(out_chs),
            nn.ReLU(inplace=True),
        )
        self.Conv3D = DownConv3D(in_chs=in_chs, out_chs=out_chs)

    def forward(self, x0, x1):
        x0 = self.up3D(x0)
        x = torch.cat((x0, x1), dim=1)
        x = self.Conv3D(x)
        return x


class Conv2D(nn.Module):
    def __init__(self, in_chs, out_chs, kernel=1, stride=1, padding=0):
        super(Conv2D, self).__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=in_chs, out_channels=out_chs, kernel_size=kernel, stride=stride, padding=padding,
                      bias=True),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv2d(x)
        return x


class U_Net_3D(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(U_Net_3D, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv0 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        self.Conv1 = DownConv3D(in_chs=16, out_chs=16)
        self.Conv2 = DownConv3D(in_chs=16, out_chs=32)
        self.Conv3 = DownConv3D(in_chs=32, out_chs=64)
        self.Conv4 = DownConv3D(in_chs=64, out_chs=128)
        self.Up3 = UpConv3D(in_chs=128, out_chs=64, output_padding=(1, 0, 0))
        self.Up2 = UpConv3D(in_chs=64, out_chs=32, output_padding=(0, 0, 0))
        self.Up1 = UpConv3D(in_chs=32, out_chs=16, output_padding=(0, 1, 1))
        self.Conv5 = DownConv3D(in_chs=16, out_chs=16)
        self.Conv1_3D = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True)
        )
        self.Conv6 = Conv2D(in_chs=in_chs, out_chs=64)
        self.Conv7 = Conv2D(in_chs=64, out_chs=32, kernel=3, stride=1, padding=1)
        self.Conv8 = Conv2D(in_chs=32, out_chs=16, kernel=3, stride=1, padding=1)
        self.Conv9 = Conv2D(in_chs=16, out_chs=out_chs)
        self.Conv10 = Conv2D(in_chs=in_chs, out_chs=out_chs)

        self.ca_zero = ChannelAttention2D(in_planes=in_chs)
        self.ca_feature = ChannelAttention2D(in_planes=in_chs)

    def forward(self, zFactor, zero):
        zFactor = self.Conv0(zFactor.unsqueeze(dim=1))
        x1 = self.Conv1(zFactor)  # (batch_size, 16, 76, 49, 49)

        x2 = self.Maxpool(x1)  # (batch_size, 16, 38, 24, 24)
        x2 = self.Conv2(x2)  # (batch_size, 32, 38, 24, 24)

        x3 = self.Maxpool(x2)  # (batch_size, 32, 19, 12, 12)
        x3 = self.Conv3(x3)  # (batch_size, 64, 19, 12, 12)

        f = self.Maxpool(x3)  # (batch_size, 64, 9, 6, 6)
        f = self.Conv4(f)  # (batch_size, 128, 9, 6, 6)

        f = self.Up3(f, x3)  # (batch_size, 64, 19, 12, 12)
        f = self.Up2(f, x2)  # (batch_size, 32, 38, 24, 24)
        f = self.Up1(f, x1)  # (batch_size, 16, 76, 49, 49)
        f = self.Conv5(f)  # (batch_size, 4, 76, 49, 49)

        f = self.Conv1_3D(f)  # (batch_size, 1, 76, 24, 24)

        f = torch.squeeze(f, dim=1)  # (batch_size, 76, 24, 24)

        ca = self.ca_zero(zero) + self.ca_feature(f)

        f = f * ca

        short_cut = self.Conv10(f)
        f = self.Conv6(f)
        f = self.Conv7(f)
        f = self.Conv8(f)
        f = self.Conv9(f)
        f = short_cut + f
        f = F.softmax(f, dim=1)

        # return f
        return f, ca


slice_width = [49, 49]
slice_num = [180, 1]
batch_size = 2
confusion_matrix = np.zeros((3, 3), dtype=int)
pixel_sum = 0

cols, rows = 2, 4
figure = plt.figure(figsize=(8, rows*3))
i = 0
GPM_BB_data = LoadBBDataset('data/Ku/val', slice_width, slice_num)
data_loader = DataLoader(GPM_BB_data, batch_size=batch_size, shuffle=True, num_workers=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = U_Net_3D(76, 3).to(device)
model = torch.load('flagBB-ca.pth').to(device)
for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
    with torch.no_grad():
        flagBB = data[:, 0, ...].numpy()
        index = []
        for j in range(data.shape[0]):
            flag = flagBB[j, ...]
            if flag[flag == 2].shape[0] >= 0.15 * slice_width[0] * slice_width[1]:
                index.append(j)
        if len(index) > 1:
            data = data[index, ...]
            target = target[index, ...]
        else:
            continue

        data, target = data.to(device).float(), target.to(device)

        flag = data[:, 0, ...].long()
        zero = data[:, 1, ...].unsqueeze(dim=1).long()
        zero = torch.zeros(zero.shape[0], 76, zero.shape[-2], zero.shape[-1], dtype=torch.float32).to(
            device).scatter_(1, zero, 1)
        factor = data[:, 2:, ...].float()

        output, zer_ca = model(factor, zero)

        output = torch.argmax(output, dim=1).cpu()
        flag = flag.cpu()

    for batch in range(output.shape[0]):
        target = flag[batch, ...]
        pre = output[batch, ...]

        if target.sum() >= 0:
            i += 1
            if 1 <= i <= rows:
                figure.add_subplot(rows, cols, i*2-1)
                plt.imshow(target.T, cmap="gray")
                figure.add_subplot(rows, cols, i*2)
                plt.imshow(pre.T, cmap="gray")
        if i >= rows:
            plt.show()
            break

#     output = output.cpu().numpy()
#     target = flag.cpu().numpy()
#     for batch in range(output.shape[0]):
#         for i in range(output.shape[1]):
#             for j in range(output.shape[2]):
#                 pred_value = output[batch][i][j]
#                 true_value = target[batch][i][j]
#                 confusion_matrix[true_value][pred_value] += 1
#     pixel_sum += (output.shape[0]*output.shape[1]*output.shape[2])
#
# print(confusion_matrix)
