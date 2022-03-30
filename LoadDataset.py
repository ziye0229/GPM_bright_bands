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
        self.slice_num = slice_num

        self.files = self.getFileList()
        self.file_index = -1

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.files) * self.slice_total

    def __init_index__(self, factor_len):
        self.index_list = []
        step = math.ceil(factor_len / self.slice_num)
        for i in range(0, self.slice_num):
            if i * step + self.slice_width <= factor_len:
                i_begin = i * step
                i_end = i * step + self.slice_width
            else:
                i_begin = factor_len - self.slice_width
                i_end = factor_len
            self.index_list.append([i_begin, i_end])

    def __getitem__(self, idx):
        if self.file_index != idx // self.slice_num:
            self.file_index = idx // self.slice_num
            file_path = self.files[self.file_index]

            with h5py.File(file_path, 'r') as f:
                if 'oversample' in file_path:
                    zFactorMeasured = np.array(f['zFactor'][idx % self.slice_num, :, :], dtype='float32')
                    zeroDeg = np.array(f['zeroDeg'][idx % self.slice_num, :, :], dtype='int32')
                    flagBB = np.array(f['flagBB'][idx % self.slice_num, :, :], dtype='int32')
                    BBTop = np.array(f['BBTop'][idx % self.slice_num, :, :], dtype='int32')
                    BBBottom = np.array(f['BBBottom'][idx % self.slice_num, :, :], dtype='int32')
                    BBPeak = np.array(f['BBPeak'][idx % self.slice_num, :, :], dtype='int32')
                else:
                    CSF = f['FS']['CSF']
                    flagBB = CSF['flagBB']
                    BBTop = CSF['binBBTop']
                    BBBottom = CSF['binBBBottom']
                    BBPeak = CSF['binBBPeak']
                    VER = f['FS']['VER']
                    zeroDeg = VER['binZeroDeg']
                    PRE = f['FS']['PRE']
                    zFactorMeasured = PRE['zFactorMeasured']
                    elevation = PRE['elevation']

                    zFactorMeasured = np.array(zFactorMeasured, dtype='float32')
                    self.__init_index__(zFactorMeasured.shape[0])
                    indexes = self.index_list[idx % self.slice_num]

                    zFactorMeasured = zFactorMeasured[indexes[0]:indexes[1]][:][100:]
                    elevation = np.array(elevation[indexes[0]:indexes[1], :], dtype='int32') // 125
                    for i in range(elevation.shape[0]):
                        for j in range(elevation.shape[1]):
                            if elevation[i, j] > 0:
                                zFactorMeasured[i, j, -elevation[i, j]:] = 0

                    zFactorMeasured[zFactorMeasured < 12] = 0
                    zFactorMeasured[zFactorMeasured >= 65] = 65
                    zFactorMeasured = zFactorMeasured / 65

                    zeroDeg = np.array(zeroDeg[indexes[0]:indexes[1], :], dtype='int32')
                    zeroDeg[zeroDeg == 177] = 176
                    zeroDeg -= 101
                    zeroDeg[zeroDeg < 0] = 0

                    flagBB = np.array(flagBB[indexes[0]:indexes[1], :], dtype='int32')
                    flagBB[flagBB > 0] = 2
                    flagBB[flagBB == 0] = 1
                    flagBB[flagBB < 0] = 0

                    BBTop = np.array(BBTop[indexes[0]:indexes[1], :], dtype='int32')
                    BBTop -= 101
                    BBTop[BBTop == -101] = 76
                    BBTop[BBTop <= -1111] = 77
                    BBTop[BBTop < 0] = 0
                    BBBottom = np.array(BBBottom[indexes[0]:indexes[1], :], dtype='int32')
                    BBBottom -= 101
                    BBBottom[BBBottom == -101] = 76
                    BBBottom[BBBottom <= -1111] = 77
                    BBBottom[BBBottom < 0] = 0
                    BBPeak = np.array(BBPeak[indexes[0]:indexes[1], :], dtype='int32')
                    BBPeak -= 101
                    BBPeak[BBPeak == -101] = 76
                    BBPeak[BBPeak <= -1111] = 77
                    BBPeak[BBPeak < 0] = 0

        image = np.concatenate((np.expand_dims(flagBB, axis=2),
                                np.expand_dims(zeroDeg, axis=2),
                                zFactorMeasured), axis=2)

        label = np.concatenate((np.expand_dims(BBTop, axis=2),
                                np.expand_dims(BBBottom, axis=2),
                                np.expand_dims(BBPeak, axis=2)), axis=2)

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


if __name__ == '__main__':
    my_dataset = LoadBBDataset('data/sliced', 49, 162)
    length = my_dataset.__len__()
