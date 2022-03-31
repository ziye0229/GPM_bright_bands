import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from LoadDataset import LoadBBDataset
from torch import optim
import math
import numpy as np


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
        self.Conv6 = Conv2D(in_chs=in_chs, out_chs=out_chs)

    def forward(self, zFactor):
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

        zFactor = torch.squeeze(f, dim=1)  # (batch_size, 76, 24, 24)

        f = self.Conv6(zFactor)
        f = F.softmax(f, dim=1)
        return f, zFactor


if __name__ == '__main__':
    slice_width = [49, 49]
    slice_num = [180, 1]
    epoch = 15
    batch_size = 16
    lr = 0.01
    lr_unchanged = True
    loss_sum = 0
    loss_cnt = 0

    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    GPM_BB_train_data = LoadBBDataset('./data/Ku/train', slice_width, slice_num)
    train_loader = DataLoader(GPM_BB_train_data, batch_size=batch_size, shuffle=False, num_workers=0)

    model = U_Net_3D(76, 3).to(device)
    # model_dict = model.state_dict()
    # pretrained_model = torch.load('flagBB.pth').to(device)
    # params = pretrained_model.named_parameters()
    # params_dict = {}
    # begin = True
    # for name, p in params:
    #     if 'factor_sa' in name:
    #         break
    #     if begin or 'Conv0' in name:
    #         params_dict[name] = p
    #         begin = True
    # model_dict.update(params_dict)
    # model.load_state_dict(model_dict)

    # model = torch.load('flagBB-ca.pth').to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    # opt = optim.SGD(model.parameters(), lr=lr)
    # cross_entropy_loss = nn.CrossEntropyLoss()
    cross_entropy_loss = nn.CrossEntropyLoss(weight=torch.tensor([1, 20, 100], dtype=torch.float32).to(device))
    focal_loss = FocalLoss(num_class=3)
    mse_loss = nn.MSELoss(reduction='mean')

    for epoch_idx in range(0, epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            # flagBB = data[:, 0, ...].numpy()
            # index = []
            # for j in range(data.shape[0]):
            #     flag = flagBB[j, ...]
            #     if flag[flag == 0].shape[0] < slice_width[0] * slice_width[1]:
            #         index.append(j)
            # if len(index) > 1:
            #     data = data[index, ...]
            #     target = target[index, ...]
            # else:
            #     continue

            data = data.to(device).float()

            with torch.no_grad():
                flag = data[:, 0, ...].long()
                zero = data[:, 1, ...].unsqueeze(dim=1).long()
                zero = torch.zeros(zero.shape[0], 76, zero.shape[-2], zero.shape[-1], dtype=torch.float32).to(
                    device).scatter_(1, zero, 1)
                zero_ca = np.zeros((data.shape[0], 76))
                for i in range(data.shape[0]):
                    zero_mean = data[i, 1, ...].mean()
                    norm = lambda x: math.exp(-(x - (zero_mean - 2)) ** 2 / (2 * 15 ** 2))
                    zero_ca[i] = np.array([norm(j) for j in range(76)])
                zero_ca = torch.from_numpy(zero_ca).unsqueeze(dim=-1).unsqueeze(dim=-1).to(device).float()
                factor = data[:, 2:, ...].float()

            # output = model(factor, zero)
            # opt.zero_grad()
            # loss = focal_loss(output, flag.long())

            output, model_zero_ca = model(factor, zero)
            opt.zero_grad()
            loss = focal_loss(output, flag.long()) + mse_loss(model_zero_ca, zero_ca)

            loss.backward()
            opt.step()

            with torch.no_grad():
                loss_sum += cross_entropy_loss(output, flag.long())
                loss_cnt += 1

                if loss_cnt == 100:
                    if lr > 0.001 and epoch_idx >= 5:
                        lr = 0.001
                        print('Change lr to {}'.format(lr))
                        opt = optim.SGD(model.parameters(), lr=lr)
                    elif lr > 0.0001 and epoch_idx >= 10:
                        lr = 0.0001
                        print('Change lr to {}'.format(lr))
                        opt = optim.SGD(model.parameters(), lr=lr)

                    Conv1_weight = model.Conv1.down3D[0].weight
                    Conv1_3D_weight = model.Conv1_3D[0].weight
                    print('Conv1 weight:\t\tMax:\t{}, Min:\t{}'.format(Conv1_weight.data.abs().max(),
                                                                       Conv1_weight.data.abs().min()))
                    print('Conv1_3D weight:\tMax:\t{}, Min:\t{}'.format(Conv1_3D_weight.data.abs().max(),
                                                                        Conv1_3D_weight.data.abs().min()))
                    print('Conv1 grad:\t\t\tMax:\t{}, Min:\t{}'.format(Conv1_weight.grad.data.abs().max(),
                                                                       Conv1_weight.grad.data.abs().min()))
                    print('Conv1_3D grad:\t\tMax:\t{}, Min:\t{}'.format(Conv1_3D_weight.grad.data.abs().max(),
                                                                        Conv1_3D_weight.grad.data.abs().min()))

                    print('epoch:{}, batch:{}, loss:{}'.format(epoch_idx, batch_idx, loss_sum / loss_cnt))
                    loss_sum = 0
                    loss_cnt = 0

        # torch.save(model, 'flagBB-epoch{}.pth'.format(epoch_idx + 1))
        torch.save(model, 'flagBB-ca.pth')
