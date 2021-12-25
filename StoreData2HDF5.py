import h5py
import numpy as np
import os
import math
import re


class Slice:
    def __init__(self, folder_dir, slice_width, slice_num, file_time=''):
        self.root_dir = folder_dir
        self.slice_width = slice_width
        self.slice_num = slice_num
        self.slice_total = slice_num[0] * slice_num[1]
        self.file_time = file_time

        self.files = self.getFileList()
        self.data_dir = {'zFactor': [], 'zeroDeg': [], 'flagBB': [], 'bbTop': [], 'bbBottom': [], 'bbPeak': []}

    def __len__(self):
        return len(self.files)

    def slice_file(self, file_idx):
        file_path = self.files[file_idx]
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

            self.zFactorMeasured_sliced = self.sliceData_3D(np.array(zFactorMeasured[:, :, 100:], dtype='float32'))
            self.zFactorMeasured_sliced[self.zFactorMeasured_sliced <= -200.0] = -200.0

            self.zeroDeg_sliced = self.sliceData_2D(np.array(ZeroDeg, dtype='int32'))
            self.zeroDeg_sliced[self.zeroDeg_sliced == 177] = 176
            self.zeroDeg_sliced -= 101

            self.flagBB_sliced = self.sliceData_2D(np.array(flagBB, dtype='int32'))
            self.flagBB_sliced[self.flagBB_sliced > 0] = 2
            self.flagBB_sliced[self.flagBB_sliced == 0] = 1
            self.flagBB_sliced[self.flagBB_sliced < 0] = 0

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

            dir = {'zFactor': self.zFactorMeasured_sliced, 'zeroDeg': self.zeroDeg_sliced, 'flagBB': self.flagBB_sliced,
                   'bbTop': self.bbTop_sliced, 'bbBottom': self.bbBottom_sliced, 'bbPeak': self.bbPeak_sliced}

        return dir

    def getFileList(self):
        hdf5_files = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if os.path.splitext(file)[1] == '.HDF5' and re.match('2A.GPM.DPR.V8-20180723.' + self.file_time, os.path.splitext(file)[0]):
                    print(os.path.splitext(file)[0])
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


slice = Slice('F:/2ADPR', [24, 24], [340, 3], file_time='201712')
dir = slice.slice_file(0)
with h5py.File("BrightBands.hdf5", "w") as f:
    for key, value in dir.items():
        f.create_dataset(key, data=value)