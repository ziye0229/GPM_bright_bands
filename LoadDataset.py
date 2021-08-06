import os
import h5py
import math
from torch.utils.data import Dataset
import numpy as np
import random


class LoadBBDataset(Dataset):
    def __init__(self, GPM_dir, slice_width, slice_num, transform=None, target_transform=None):
        self.root_dir = GPM_dir
        self.slice_width = slice_width
        self.slice_num = slice_num

        self.files = self.getFileList()
        self.file_index = -1

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.files) * self.slice_num

    def __getitem__(self, idx):
        if self.file_index != idx // self.slice_num:
            self.file_index = idx // self.slice_num
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

                self.zeroDeg_sliced = self.sliceData_2D(np.array(ZeroDeg, dtype='int32'))
                self.zFactorMeasured_sliced = self.sliceData_3D(np.array(zFactorMeasured, dtype='float32'))
                self.zFactorMeasured_sliced[self.zFactorMeasured_sliced <= -200.0] = -200.0

                self.flagBB_sliced = self.sliceData_2D(np.array(flagBB, dtype='int32'))
                self.flagBB_sliced[self.flagBB_sliced < 0] = 0
                self.flagBB_sliced[self.flagBB_sliced == 0] = 1
                self.flagBB_sliced[self.flagBB_sliced == 1] = 2
                self.flagBB_sliced[self.flagBB_sliced == 2] = 3
                self.flagBB_sliced[self.flagBB_sliced == 3] = 4
                self.bbTop_sliced = self.sliceData_2D(np.array(BBTop, dtype='int16'))
                self.bbTop_sliced[self.bbTop_sliced <= 0] = 0
                self.bbBottom_sliced = self.sliceData_2D(np.array(BBBottom, dtype='int16'))
                self.bbBottom_sliced[self.bbBottom_sliced <= 0] = 0
                self.bbPeak_sliced = self.sliceData_2D(np.array(BBPeak, dtype='int16'))
                self.bbPeak_sliced[self.bbPeak_sliced <= 0] = 0

        image = self.getImage(idx)
        label = self.getLable(idx)

        # label_sum = label.sum()
        # print(label_sum)
        # 有问题
        # while label.sum() == 0:
        #     idx += 1
        #     image = self.getImage(idx)
        #     label = self.getLable(idx)

        image = image.transpose(2, 0, 1)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label,

    def getFileList(self):
        hdf5_files = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if os.path.splitext(file)[1] == '.HDF5':
                    hdf5_files.append(os.path.join(root, file))
        return hdf5_files

    def sliceData_2D(self, data):
        data_np = np.empty([self.slice_num, self.slice_width, data.shape[-1]], dtype=data.dtype)
        step = math.ceil((data.shape[0] - self.slice_width) / self.slice_num)

        for i in range(0, self.slice_num):
            if i * step + self.slice_width <= data.shape[0]:
                # print(i*step, i*step + self.slice_width)
                data_np[i] = np.array(data[i * step:i * step + self.slice_width][:], dtype=data.dtype)
            else:
                # print(data.shape[0]-data_np.shape[1], data.shape[0])
                data_np[i] = np.array(data[data.shape[0] - self.slice_width:][:], dtype=data.dtype)
        return data_np

    def sliceData_3D(self, data):
        data_np = np.empty([self.slice_num, self.slice_width, data.shape[-2], data.shape[-1]], dtype=data.dtype)
        step = math.ceil((data.shape[0] - self.slice_width) / self.slice_num)

        for i in range(0, self.slice_num):
            if i * step + self.slice_width <= data.shape[0]:
                # print(i*step, i*step + self.slice_width)
                data_np[i] = np.array(data[i * step:i * step + self.slice_width][:][:], dtype=data.dtype)
            else:
                # print(data.shape[0]-data_np.shape[1], data.shape[0])
                data_np[i] = np.array(data[data.shape[0] - data_np.shape[1]:][:][:], dtype=data.dtype)
        return data_np

    def getImage(self, idx):
        return np.concatenate((np.expand_dims(self.zeroDeg_sliced[idx % self.slice_num], axis=2),
                               self.zFactorMeasured_sliced[idx % self.slice_num]), axis=2)

    def getLable(self, idx):
        # return np.concatenate((np.expand_dims(self.bbTop_sliced[idx % self.slice_num], axis=0),
        #                         np.expand_dims(self.bbBottom_sliced[idx % self.slice_num], axis=0),
        #                         np.expand_dims(self.bbPeak_sliced[idx % self.slice_num], axis=0)))

        # return self.bbTop_sliced[idx % self.slice_num]

        return self.flagBB_sliced[idx % self.slice_num]

if __name__ == '__main__':
    my_dataset = LoadBBDataset('data/train', 101, 150)
    length = my_dataset.__len__()
    label_sum = 0
    for i in random.sample(range(0, length), 100):
        result = my_dataset.__getitem__(i)

        print(result[0], result[1])
        print(result[0].shape, result[1].shape)
        
    #     result[1][result[1] > 0] = 1
    #     label_sum += result[1].sum()
    #     if (i+1) % 100 == 0:
    #         print(label_sum / ((i+1) * 101 * 49))
    # print(label_sum / (length * 101 * 49))

