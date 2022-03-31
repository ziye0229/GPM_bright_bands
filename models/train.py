import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from LoadDataset import LoadBBDataset
from torch.utils.data import RandomSampler
from torch import optim
import math
import numpy as np

from models.U_Net_3D import U_Net_3D


class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.reshape(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


slice_width = 49
slice_num = 162
epoch = 15
batch_size = 8
lr = 0.01
lr_unchanged = True
loss_sum = 0
loss_cnt = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

GPM_BB_train_data = LoadBBDataset('../data/Ku/raw_train', slice_width, slice_num)
train_loader = DataLoader(GPM_BB_train_data, batch_size=batch_size, sampler=RandomSampler(GPM_BB_train_data),
                          pin_memory=True, num_workers=4)

model = U_Net_3D(76, 78).to(device)

opt = optim.Adam(model.parameters(), lr=lr)
# opt = optim.SGD(model.parameters(), lr=lr)
focal_loss = FocalLoss(num_class=78)

for epoch_idx in range(0, epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True).long()

        with torch.no_grad():
            flag = data[:, 0, ...].long()
            # zero = data[:, 1, ...].unsqueeze(dim=1).long()
            # zero = torch.zeros(zero.shape[0], 76, zero.shape[-2], zero.shape[-1], dtype=torch.float32).to(
            #     device).scatter_(1, zero, 1)
            # zero_ca = np.zeros((data.shape[0], 76))
            # for i in range(data.shape[0]):
            #     zero_mean = data[i, 1, ...].mean()
            #     norm = lambda x: math.exp(-(x - (zero_mean - 2)) ** 2 / (2 * 15 ** 2))
            #     zero_ca[i] = np.array([norm(j) for j in range(76)])
            # zero_ca = torch.from_numpy(zero_ca).unsqueeze(dim=-1).unsqueeze(dim=-1).to(device).float()
            factor = data[:, 2:, ...]

            BBPeak = target[:, -1, ...]

        output = model(factor)
        opt.zero_grad()
        loss = focal_loss(output, BBPeak)

        loss.backward()
        opt.step()

        with torch.no_grad():
            loss_sum += loss
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
    torch.save(model, 'U_Net_3D.pth')