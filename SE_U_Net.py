import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(3, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, flag):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out, flag], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class DownConv3D(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(DownConv3D, self).__init__()
        self.down3D = nn.Sequential(
            nn.Conv3d(in_channels=in_chs, out_channels=out_chs, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_chs),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.down3D(x)
        return x


class UpConv3D(nn.Module):
    def __init__(self, in_chs, out_chs, output_padding):
        super(UpConv3D, self).__init__()
        self.up3D = nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_chs, out_channels=out_chs, kernel_size=4, padding=1, stride=2, bias=True,
                               output_padding=output_padding),
            nn.BatchNorm3d(out_chs),
            nn.ReLU(inplace=True)
        )
        self.Conv3D = DownConv3D(in_chs=in_chs, out_chs=out_chs)

    def forward(self, x0, x1):
        x0 = self.up3D(x0)
        x = torch.cat((x0, x1), dim=1)
        x = self.Conv3D(x)
        return x


class SE_U_Net(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(SE_U_Net, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

        self.CA1 = ChannelAttention(in_chs, ratio=4)
        self.Conv1 = DownConv3D(in_chs=1, out_chs=16)
        self.Conv2 = DownConv3D(in_chs=16, out_chs=32)
        self.Conv3 = DownConv3D(in_chs=32, out_chs=64)
        self.Conv4 = DownConv3D(in_chs=64, out_chs=128)
        self.Up3 = UpConv3D(in_chs=128, out_chs=64, output_padding=(1, 0, 0))
        self.Up2 = UpConv3D(in_chs=64, out_chs=32, output_padding=(0, 0, 0))
        self.Up1 = UpConv3D(in_chs=32, out_chs=16, output_padding=(0, 0, 0))
        self.Conv5 = DownConv3D(in_chs=16, out_chs=4)
        self.Conv1_3D = nn.Sequential(
            nn.Conv3d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True)
        )
        self.CA2 = ChannelAttention(in_chs, ratio=4)
        self.SA = SpatialAttention(kernel_size=1)

    def forward(self, flagBB, zeroDeg, zFactor):
        zFactor = zFactor * self.CA1(zFactor)
        x1 = self.Conv1(zFactor.unsqueeze(dim=1))  # (batch_size, 64, 76, 24, 24)

        x2 = self.Maxpool(x1)  # (batch_size, 64, 38, 12, 12)
        x2 = self.Conv2(x2)  # (batch_size, 128, 38, 12, 12)

        x3 = self.Maxpool(x2)  # (batch_size, 128, 19, 6, 6)
        x3 = self.Conv3(x3)  # (batch_size, 256, 19, 6, 6)

        f = self.Maxpool(x3)  # (batch_size, 256, 9, 3, 3)
        f = self.Conv4(f)  # (batch_size, 512, 9, 3, 3)

        f = self.Up3(f, x3)  # (batch_size, 256, 44, 6, 6)
        f = self.Up2(f, x2)  # (batch_size, 128, 88, 12, 12)
        f = self.Up1(f, x1)  # (batch_size, 64, 176, 24, 24)
        f = self.Conv5(f)  # (batch_size, 4, 176, 24, 24)

        f = self.Conv1_3D(torch.cat((f, zeroDeg.unsqueeze(dim=1)), dim=1))  # (batch_size, 1, 76, 101, 49)
        f = torch.squeeze(f, dim=1)  # (batch_size, 76, 101, 49)

        f = f * self.CA2(f)
        f = f * self.SA(f, flagBB[:, 2, :, :].unsqueeze(dim=1))

        f = torch.cat((f, flagBB[:, 1, :, :].unsqueeze(dim=1), flagBB[:, 0, :, :].unsqueeze(dim=1)), dim=1)  # (batch_size 78, 101, 49)

        f = nn.functional.softmax(f, dim=1)
        return f

