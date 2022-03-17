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
batch_size = 64
lr = 0.0005
lr_unchanged = True
loss_sum = 0
loss_cnt = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

GPM_BB_train_data = LoadBBDataset('./data/sliced', slice_width, slice_num)
train_loader = DataLoader(GPM_BB_train_data, batch_size=batch_size, shuffle=False, num_workers=0)

# model = SE_U_Net(76, 1).to(device)
model = torch.load('BBPeak-epoch15.pth').to(device)
# model_dict = model.state_dict()
# pretrained_model = torch.load('BBPeak-epoch2.pth').to(device)
# params = pretrained_model.named_parameters()
# params_dict = {}
# begin = False
# for name, p in params:
#     if 'CA' in name:
#         break
#     if begin or 'Conv1.down' in name:
#         params_dict[name] = p
#         begin = True
# model_dict.update(params_dict)
# model.load_state_dict(model_dict)

opt = optim.SGD(model.parameters(), lr=lr)
# opt = optim.Adam(model.parameters(), lr=lr)

Loss = nn.CrossEntropyLoss(reduction='mean')

for i in range(0, epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
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

        flag = data[:, 0, ...].unsqueeze(dim=1).long()
        flag = torch.zeros(flag.shape[0], 3, flag.shape[-2], flag.shape[-1], dtype=torch.float32).to(device).scatter_(1, flag, 1)
        zero = data[:, 1, ...].unsqueeze(dim=1).long()
        # zero = torch.zeros(zero.shape[0], 76, zero.shape[-2], zero.shape[-1], dtype=torch.float32).to(device).scatter_(1, zero, 1)
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
        loss_cnt += 1
        times = 1

        if loss_cnt == 100:
            if lr_unchanged and loss_sum/loss_cnt < 3:
                print('Change lr to {}'.format(lr / 5))
                opt = optim.SGD(model.parameters(), lr=lr / 5)
                lr_unchanged = False

            Conv1_weight = model.Conv1.down3D[0].weight
            Conv1_3D_weight = model.Conv1_3D[0].weight
            print('Conv1 weight:\t\tMax:\t{}, Min:\t{}'.format(Conv1_weight.data.abs().max(), Conv1_weight.data.abs().min()))
            print('Conv1_3D weight:\tMax:\t{}, Min:\t{}'.format(Conv1_3D_weight.data.abs().max(), Conv1_3D_weight.data.abs().min()))
            print('Conv1 grad:\t\t\tMax:\t{}, Min:\t{}'.format(Conv1_weight.grad.data.abs().max(), Conv1_weight.grad.data.abs().min()))
            print('Conv1_3D grad:\t\tMax:\t{}, Min:\t{}'.format(Conv1_3D_weight.grad.data.abs().max(),
                                                                Conv1_3D_weight.grad.data.abs().min()))

            print('epoch:{}, batch:{}, loss:{}'.format(i + 1, batch_idx, loss_sum/loss_cnt))
            loss_sum = 0
            loss_cnt = 0

    print(peak[0, ...].view(24 * 24))
    print(output.argmax(dim=1)[0, ...].view(24 * 24))
    torch.save(model, 'BBPeak-epoch{}.pth'.format(i + 1))