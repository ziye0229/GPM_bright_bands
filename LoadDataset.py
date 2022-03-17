import os
import h5py
import math
from torch.utils.data import Dataset
import numpy as np
import random
from collections import Counter


class LoadBBDataset(Dataset):
    def __init__(self, GPM_dir, slice_width, slice_num, transform=None, target_transform=None):
        self.root_dir = GPM_dir
        self.slice_width = slice_width
        self.slice_total = slice_num[0] * slice_num[1]

        self.files = self.getFileList()
        self.file_index = -1

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.files) * self.slice_total

    def __getitem__(self, idx):
        if self.file_index != idx // self.slice_total:
            self.file_index = idx // self.slice_total
            file_path = self.files[self.file_index]
            with h5py.File(file_path, 'r') as f:
                self.zFactorMeasured_sliced = np.array(f['zFactor'], dtype='float32')
                self.zeroDeg_sliced = np.array(f['zeroDeg'], dtype='int32')
                self.flagBB_sliced = np.array(f['flagBB'], dtype='int32')
                self.bbTop_sliced = np.array(f['bbTop'], dtype='int32')
                self.bbBottom_sliced = np.array(f['bbBottom'], dtype='int32')
                self.bbPeak_sliced = np.array(f['bbPeak'], dtype='int32')

        image = np.concatenate((np.expand_dims(self.flagBB_sliced[idx % self.slice_total], axis=2),
                                np.expand_dims(self.zeroDeg_sliced[idx % self.slice_total], axis=2),
                                self.zFactorMeasured_sliced[idx % self.slice_total]), axis=2)

        label = np.concatenate((np.expand_dims(self.bbTop_sliced[idx % self.slice_total], axis=2),
                                np.expand_dims(self.bbBottom_sliced[idx % self.slice_total], axis=2),
                                np.expand_dims(self.bbPeak_sliced[idx % self.slice_total], axis=2)), axis=2)

        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def getFileList(self):
        hdf5_files = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if os.path.splitext(file)[1] == '.HDF5':
                    hdf5_files.append(os.path.join(root, file))
        return hdf5_files

    def getImage(self, idx):
        return np.concatenate((np.expand_dims(self.zeroDeg_sliced[idx % self.slice_total], axis=2),
                               self.zFactorMeasured_sliced[idx % self.slice_total]), axis=2)

    def getLable(self, idx):
        return self.flagBB_sliced[idx % self.slice_total]


if __name__ == '__main__':
    my_dataset = LoadBBDataset('data/sliced', [49, 49], [180, 1])
    length = my_dataset.__len__()
    # label_sum = 0
    cnt = Counter()
    for i in range(0, 340 * 3 * 20):
        data, label = my_dataset.__getitem__(i)
        flagBB = data[0, ...]
        if flagBB[flagBB == 0].shape[0] < 24 * 24:
            # if True:
            for flags in flagBB.tolist():
                cnt += Counter(flags)
    total = 0
    for i in cnt:
        total += cnt[i]
    for i in cnt:
        print(cnt[i] / total)

    #     result[1][result[1] > 0] = 1
    #     label_sum += result[1].sum()
    #     if (i+1) % 100 == 0:
    #         print(label_sum / ((i+1) * 101 * 49))
    # print(label_sum / (length * 101 * 49))
