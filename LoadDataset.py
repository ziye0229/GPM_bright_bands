import os
import h5py
import math
from torch.utils.data import Dataset
import numpy as np
import random
from collections import Counter


class LoadBBDataset(Dataset):
    def __init__(self, GPM_dir, slice_width, slice_num, data_type='flagBB', transform=None, target_transform=None):
        self.root_dir = GPM_dir
        self.slice_width = slice_width
        self.slice_num = slice_num
        self.slice_total = slice_num[0] * slice_num[1]
        self.data_type = data_type

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
            # print('Loading {}/{}\t{}'.format(idx // self.slice_num, len(self.files), file_path.split('\\')[-1]))
            with h5py.File(file_path, 'r') as f:
                CSF = f['NS']['CSF']
                flagBB = CSF['flagBB']
                BBTop = CSF['binBBTop']
                BBBottom = CSF['binBBBottom']
                BBPeak = CSF['binBBPeak']
                VER = f['NS']['VER']
                ZeroDeg = VER['binZeroDeg']
                PRE = f['NS']['PRE']
                zFactorMeasured = PRE['zFactorMeasured']

                if self.data_type == 'flagBB':
                    self.zFactorMeasured_sliced = self.sliceData_3D(np.array(zFactorMeasured[:, :, :], dtype='float32'))
                elif self.data_type == 'BBPosition':
                    self.zFactorMeasured_sliced = self.sliceData_3D(
                        np.array(zFactorMeasured[:, :, 100:], dtype='float32'))
                self.zFactorMeasured_sliced[self.zFactorMeasured_sliced <= -200.0] = -200.0

                self.zeroDeg_sliced = self.sliceData_2D(np.array(ZeroDeg, dtype='int32'))
                if self.data_type == 'BBPosition':
                    self.zeroDeg_sliced[self.zeroDeg_sliced == 177] = 176
                    self.zeroDeg_sliced -= 101

                self.flagBB_sliced = self.sliceData_2D(np.array(flagBB, dtype='int32'))
                self.flagBB_sliced[self.flagBB_sliced > 0] = 2
                self.flagBB_sliced[self.flagBB_sliced == 0] = 1
                self.flagBB_sliced[self.flagBB_sliced < 0] = 0

                if self.data_type == 'flagBB':
                    self.bbTop_sliced = self.sliceData_2D(np.array(BBTop, dtype='int32'))
                    self.bbTop_sliced[self.bbTop_sliced <= 0] = 0
                    self.bbBottom_sliced = self.sliceData_2D(np.array(BBBottom, dtype='int32'))
                    self.bbBottom_sliced[self.bbBottom_sliced <= 0] = 0
                    self.bbPeak_sliced = self.sliceData_2D(np.array(BBPeak, dtype='int32'))
                    self.bbPeak_sliced[self.bbPeak_sliced <= 0] = 0
                elif self.data_type == 'BBPosition':
                    self.bbTop_sliced = self.sliceData_2D(np.array(BBTop, dtype='int32'))
                    self.bbTop_sliced -= 101
                    self.bbTop_sliced[self.bbTop_sliced == -101] = 76
                    self.bbTop_sliced[self.bbTop_sliced <= -1111] = 77
                    self.bbBottom_sliced = self.sliceData_2D(np.array(BBBottom, dtype='int32'))
                    self.bbBottom_sliced -= 101
                    self.bbBottom_sliced[self.bbBottom_sliced == -101] = 76
                    self.bbBottom_sliced[self.bbBottom_sliced <= -1111] = 77
                    self.bbPeak_sliced = self.sliceData_2D(np.array(BBPeak, dtype='int32'))
                    self.bbPeak_sliced -= 101
                    self.bbPeak_sliced[self.bbPeak_sliced == -101] = 76
                    self.bbPeak_sliced[self.bbPeak_sliced <= -1111] = 77

        image = self.getImage(idx)
        label = self.getLable(idx)

        image = image.transpose(2, 0, 1)
        if self.data_type == 'BBPosition':
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

    def sliceData_2D(self, data):
        data_np = np.empty([self.slice_total, self.slice_width[0], self.slice_width[1]], dtype=data.dtype)
        step = [-1, -1]
        step[0] = math.ceil(data.shape[0] / self.slice_num[0])
        step[1] = math.ceil(data.shape[1] / self.slice_num[1])

        for i in range(0, self.slice_num[0]):
            if i * step[0] + self.slice_width[0] <= data.shape[0]:
                i_begin = i * step[0]
                i_end = i * step[0] + self.slice_width[0]
            else:
                i_begin = data.shape[0] - self.slice_width[0]
                i_end = data.shape[0]+1
            # print(i_begin, i_end)
            for j in range(0, self.slice_num[1]):
                if j * step[1] + self.slice_width[1] <= data.shape[1]:
                    j_begin = j*step[1]
                    j_end = j*step[1] + self.slice_width[1]
                else:
                    j_begin = data.shape[1] - self.slice_width[1]
                    j_end = data.shape[1]+1
                # print('\t', j_begin, j_end)
                data_np[i * self.slice_num[1] + j] = np.array(data[i_begin:i_end, j_begin:j_end], dtype=data.dtype)
        return data_np

    def sliceData_3D(self, data):
        data_np = np.empty([self.slice_total, self.slice_width[0], self.slice_width[1], data.shape[-1]], dtype=data.dtype)
        step = [-1, -1]
        step[0] = math.ceil(data.shape[0] / self.slice_num[0])
        step[1] = math.ceil(data.shape[1] / self.slice_num[1])

        for i in range(0, self.slice_num[0]):
            if i * step[0] + self.slice_width[0] <= data.shape[0]:
                i_begin = i * step[0]
                i_end = i * step[0] + self.slice_width[0]
            else:
                i_begin = data.shape[0] - self.slice_width[0]
                i_end = data.shape[0]+1
            for j in range(0, self.slice_num[1]):
                if j * step[1] + self.slice_width[1] <= data.shape[1]:
                    j_begin = j*step[1]
                    j_end = j*step[1] + self.slice_width[1]
                else:
                    j_begin = data.shape[1] - self.slice_width[1]
                    j_end = data.shape[1]+1
                data_np[i * self.slice_num[1] + j] = np.array(data[i_begin:i_end, j_begin:j_end][:], dtype=data.dtype)
        return data_np

    def getImage(self, idx):
        if self.data_type == 'flagBB':
            return np.concatenate((np.expand_dims(self.zeroDeg_sliced[idx % self.slice_total], axis=2),
                                   self.zFactorMeasured_sliced[idx % self.slice_total]), axis=2)
        elif self.data_type == 'BBPosition':
            return np.concatenate((np.expand_dims(self.flagBB_sliced[idx % self.slice_total], axis=2),
                                   np.expand_dims(self.zeroDeg_sliced[idx % self.slice_total], axis=2),
                                   self.zFactorMeasured_sliced[idx % self.slice_total]), axis=2)

    def getLable(self, idx):
        if self.data_type == 'flagBB':
            return self.flagBB_sliced[idx % self.slice_total]
        elif self.data_type == 'BBPosition':
            return np.concatenate((np.expand_dims(self.bbTop_sliced[idx % self.slice_total], axis=2),
                                   np.expand_dims(self.bbBottom_sliced[idx % self.slice_total], axis=2),
                                   np.expand_dims(self.bbPeak_sliced[idx % self.slice_total], axis=2)), axis=2)


if __name__ == '__main__':
    my_dataset = LoadBBDataset('data/train', [24, 24], [340, 3], data_type='flagBB')
    length = my_dataset.__len__()
    # label_sum = 0
    cnt = Counter()
    for i in range(0, 340*3*5):
        data, label = my_dataset.__getitem__(i)
        if label[label == 2].shape[0] > 0.1 * 24 * 24:
        # if True:
            for flags in label.tolist():
                cnt += Counter(flags)
    total = 0
    for i in cnt:
        total += cnt[i]
    for i in cnt:
        print(cnt[i]/total)

    #     result[1][result[1] > 0] = 1
    #     label_sum += result[1].sum()
    #     if (i+1) % 100 == 0:
    #         print(label_sum / ((i+1) * 101 * 49))
    # print(label_sum / (length * 101 * 49))
