import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from LoadDataset import LoadBBDataset
from torch import optim
import math

from SE_U_Net import SE_U_Net


slice_width = [24, 24]
slice_num = [340, 3]
epoch = 20
batch_size = 12
lr = 0.001
loss_sum = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

GPM_BB_train_data = LoadBBDataset('data/train', slice_width, slice_num, data_type='BBPosition')
train_loader = DataLoader(GPM_BB_train_data, batch_size=batch_size, shuffle=False, num_workers=0)

model = SE_U_Net(76, 1).to(device)
# model = torch.load('model-epoch10-batch4000.pth').to(device)

# opt = optim.SGD(model.parameters(), lr=lr)
opt = optim.Adam(model.parameters(), lr=lr)

Loss = nn.CrossEntropyLoss()

for i in range(0, epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        flagBB = data[:, 0, ...].numpy()
        if flagBB[flagBB == 2].shape[0] < 0.25 * slice_width[0] * slice_width[1] * batch_size:
            continue

        data, target = data.to(device).float(), target.to(device)

        flag = data[:, 0, ...].unsqueeze(dim=1).long()
        flag = torch.zeros(flag.shape[0], 3, flag.shape[-2], flag.shape[-1], dtype=torch.float32).to(device).scatter_(1, flag, 1)
        zero = data[:, 1, ...].unsqueeze(dim=1).long()
        zero = torch.zeros(zero.shape[0], 76, zero.shape[-2], zero.shape[-1], dtype=torch.float32).to(device).scatter_(1, zero, 1)
        factor = data[:, 2:, ...].float()
        top = target[:, 0, ...].long()
        bottom = target[:, 1, ...].long()
        peak = target[:, 2, ...].long()

        output = model(flag, zero, factor)

        opt.zero_grad()
        loss = Loss(output, peak)
        loss.backward()

        opt.step()

        loss_sum += loss
        loss_sum += math.exp(loss)
        times = 1
        if batch_idx % times == 0:
            Conv1_weight = model.Conv1.down3D[0].weight
            Conv1_3D_weight = model.Conv1_3D[0].weight
            print('Conv1 weight:\t\tMax:\t{}, Min:\t{}'.format(Conv1_weight.data.abs().max(), Conv1_weight.data.abs().min()))
            print('Conv1_3D weight:\tMax:\t{}, Min:\t{}'.format(Conv1_3D_weight.data.abs().max(), Conv1_3D_weight.data.abs().min()))
            print('Conv1 grad:\t\t\tMax:\t{}, Min:\t{}'.format(Conv1_weight.grad.data.abs().max(), Conv1_weight.grad.data.abs().min()))
            print('Conv1_3D grad:\t\tMax:\t{}, Min:\t{}'.format(Conv1_3D_weight.grad.data.abs().max(),
                                                                Conv1_3D_weight.grad.data.abs().min()))

            print('epoch:{}, batch:{}, loss:{}'.format(i + 1, batch_idx, loss_sum/times))
            loss_sum = 0

    torch.save(model, 'BBPeak-epoch{}.pth'.format(i + 1))