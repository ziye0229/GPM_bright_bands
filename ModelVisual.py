import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from LoadDataset import LoadBBDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from U_Net import *


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
        U_Net_out = f

        ca = self.ca_zero(zero) + self.ca_feature(f)

        f = f * ca
        ca_out = f

        short_cut = self.Conv10(f)
        f = self.Conv6(f)
        f = self.Conv7(f)
        f = self.Conv8(f)
        f = self.Conv9(f)
        f = short_cut + f
        f = F.softmax(f, dim=1)

        return f, ca, U_Net_out, ca_out

def draw_bar(x, y, title=''):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(y=x, width=y, height=1)
    ax.set_title(title, fontsize=15)
    plt.show()

def matix_density(mat, dence=0.2):
    x = [i for i in range(0, mat.shape[0])]
    y = []
    for index in x:
        slice = mat[index]
        y.append(slice[abs(slice) > dence].shape[0])
    return x, y


slice_width = [49, 49]
slice_num = [180, 1]
batch_size = 2

GPM_BB_data = LoadBBDataset('data/Ku/val', slice_width, slice_num)
data_loader = DataLoader(GPM_BB_data, batch_size=batch_size, shuffle=True, num_workers=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('flagBB-ca.pth').to(device)
for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
    with torch.no_grad():
        data, target = data.to(device).float(), target.to(device)

        flag = data[:, 0, ...].long()
        zero = data[:, 1, ...].unsqueeze(dim=1).long()
        zero = torch.zeros(zero.shape[0], 76, zero.shape[-2], zero.shape[-1], dtype=torch.float32).to(
            device).scatter_(1, zero, 1)
        factor = data[:, 2:, ...].float()

        output, zero_ca, U_Net_out, ca_out = model(factor, zero)

        for i in range(factor.shape[0]):
            flagBB = flag[i, ...]
            if flagBB[flagBB == 2].shape[0] >= 0.15 * slice_width[0] * slice_width[1] and data[i, 1, ...].mean() < 70:
                print(data[i, 1, ...].mean())
                factor = factor[i, ...].to('cpu').numpy()
                x, y = matix_density(factor, dence=0)
                draw_bar(x, y, 'Input distribution')
                U_Net_out = U_Net_out[i, ...].to('cpu').numpy()
                x, y = matix_density(U_Net_out)
                draw_bar(x, y, 'After U_Net distribution')
                zero_ca = zero_ca[i, ...].to('cpu').numpy()
                x = [index for index in range(zero_ca.shape[0])]
                y = [float(zero_ca[index]) for index in x]
                draw_bar(x, y, 'Attention distribution')
                ca_out = ca_out[i, ...].to('cpu').numpy()
                x, y = matix_density(ca_out)
                draw_bar(x, y, 'After attention distribution')
                break