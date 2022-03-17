import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from LoadDataset import LoadBBDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from SE_U_Net import SE_U_Net


slice_width = [24, 24]
slice_num = [340, 3]
batch_size = 32
confusion_matrix = np.zeros((3, 3), dtype=int)
pixel_sum = 0

cols, rows = 2, 6
figure = plt.figure(figsize=(8, rows*3))
i = 0
GPM_BB_data = LoadBBDataset('data/val', slice_width, slice_num)
data_loader = DataLoader(GPM_BB_data, batch_size=batch_size, shuffle=True, num_workers=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.load('BBPeak-epoch15.pth').to(device)
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

        flag = data[:, 0, ...].unsqueeze(dim=1).long()
        flag = torch.zeros(flag.shape[0], 3, flag.shape[-2], flag.shape[-1], dtype=torch.float32).to(device).scatter_(1,
                                                                                                                      flag,
                                                                                                                      1)
        zero = data[:, 1, ...].unsqueeze(dim=1).long()
        # zero = torch.zeros(zero.shape[0], 76, zero.shape[-2], zero.shape[-1], dtype=torch.float32).to(device).scatter_(1, zero, 1)
        factor = data[:, 2:, ...].float()
        top = target[:, 0, ...].long()
        bottom = target[:, 1, ...].long()
        peak = target[:, 2, ...].long()

        output = model(flag, zero, factor)

        output = F.softmax(output)
        output = output * flag[:, 2, :, :].unsqueeze(dim=1)
        output[:, -2, ...] = flag[:, 1, :, :]
        output[:, -1, ...] = flag[:, 0, :, :]

        output = torch.argmax(output, dim=1).cpu()
        position = peak.cpu()

    for batch in range(output.shape[0]):
        target = position[batch, ...]
        pre = output[batch, ...]

        if target.sum() >= 0:
            i += 1
            if 1 <= i <= rows:
                figure.add_subplot(rows, cols, i*2-1)
                plt.imshow(target.T, cmap="gray")
                plt.colorbar()
                figure.add_subplot(rows, cols, i*2)
                plt.imshow((target-pre).abs().T, cmap="gray")
                plt.colorbar()
        if i >= rows:
            plt.show()
            break

    # output = output.cpu().numpy()
    # target = flag.cpu().numpy()
    # for batch in range(output.shape[0]):
    #     for i in range(output.shape[1]):
    #         for j in range(output.shape[2]):
    #             pred_value = output[batch][i][j]
    #             true_value = target[batch][i][j]
    #             confusion_matrix[true_value][pred_value] += 1
    # pixel_sum += (output.shape[0]*output.shape[1]*output.shape[2])

print(confusion_matrix)
