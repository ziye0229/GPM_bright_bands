import h5py
import numpy as np
import os
import math
import re
from tqdm import tqdm


class Slice:
    def __init__(self, s_folder_dir, d_folder_dir, slice_width, slice_num, file_time=''):
        self.root_dir = s_folder_dir
        self.des = d_folder_dir
        self.slice_width = slice_width
        self.slice_num = slice_num
        self.slice_total = slice_num[0] * slice_num[1]
        self.file_time = file_time

        self.files = self.get_file_list()

    def __len__(self):
        return len(self.files)

    def store_HDF5(self):
        for i in tqdm(range(len(self.files))):
            data = self.slice_file(i)
            file_name = os.path.splitext(self.files[i].split('/')[-1])[0].split('.')[4]
            with h5py.File(self.des + '/{}.HDF5'.format(file_name), "w") as f:
                for key, value in data.items():
                    f.create_dataset(key, data=value)


    def slice_file(self, file_idx):
        file_path = self.files[file_idx]
        with h5py.File(file_path, 'r') as f:
            CSF = f['FS']['CSF']
            flagBB = CSF['flagBB']
            BBTop = CSF['binBBTop']
            BBBottom = CSF['binBBBottom']
            BBPeak = CSF['binBBPeak']
            VER = f['FS']['VER']
            ZeroDeg = VER['binZeroDeg']
            PRE = f['FS']['PRE']
            zFactorMeasured = PRE['zFactorMeasured']
            elevation = PRE['elevation']

            zFactorMeasured = np.array(zFactorMeasured[:, :, 100:], dtype='float32')
            elevation = np.array(elevation, dtype='int32') // 125
            for i in range(elevation.shape[0]):
                for j in range(elevation.shape[1]):
                    if elevation[i, j] > 0:
                        zFactorMeasured[i, j, -elevation[i, j]:] = 0

            zFactorMeasured_sliced = self.slice_data_3D(zFactorMeasured)
            zFactorMeasured_sliced[zFactorMeasured_sliced < 12] = 0
            zFactorMeasured_sliced[zFactorMeasured_sliced >= 65] = 65
            zFactorMeasured_sliced = zFactorMeasured_sliced / 65

            zeroDeg_sliced = self.slice_data_2D(np.array(ZeroDeg, dtype='int32'))
            zeroDeg_sliced[zeroDeg_sliced == 177] = 176
            zeroDeg_sliced -= 101
            zeroDeg_sliced[zeroDeg_sliced < 0] = 0

            flagBB_sliced = self.slice_data_2D(np.array(flagBB, dtype='int32'))
            flagBB_sliced[flagBB_sliced > 0] = 2
            flagBB_sliced[flagBB_sliced == 0] = 1
            flagBB_sliced[flagBB_sliced < 0] = 0

            bbTop_sliced = self.slice_data_2D(np.array(BBTop, dtype='int32'))
            bbTop_sliced -= 101
            bbTop_sliced[bbTop_sliced == -101] = 76
            bbTop_sliced[bbTop_sliced <= -1111] = 77
            bbTop_sliced[bbTop_sliced < 0] = 0
            bbBottom_sliced = self.slice_data_2D(np.array(BBBottom, dtype='int32'))
            bbBottom_sliced -= 101
            bbBottom_sliced[bbBottom_sliced == -101] = 76
            bbBottom_sliced[bbBottom_sliced <= -1111] = 77
            bbBottom_sliced[bbBottom_sliced < 0] = 0
            bbPeak_sliced = self.slice_data_2D(np.array(BBPeak, dtype='int32'))
            bbPeak_sliced -= 101
            bbPeak_sliced[bbPeak_sliced == -101] = 76
            bbPeak_sliced[bbPeak_sliced <= -1111] = 77
            bbPeak_sliced[bbPeak_sliced < 0] = 0

            data_dir = {'zFactor': zFactorMeasured_sliced, 'zeroDeg': zeroDeg_sliced, 'flagBB': flagBB_sliced,
                        'bbTop': bbTop_sliced, 'bbBottom': bbBottom_sliced, 'bbPeak': bbPeak_sliced}
            return data_dir

    def get_file_list(self):
        hdf5_files = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if os.path.splitext(file)[1] == '.HDF5' and re.match('2A.GPM.Ku.V9-20211125.' + self.file_time, os.path.splitext(file)[0]):
                    # print(os.path.splitext(file)[0])
                    hdf5_files.append(os.path.join(root, file))
        return hdf5_files

    def slice_data_2D(self, data):
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

    def slice_data_3D(self, data):
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


slice = Slice('./data/Ku/raw_val', './data/Ku/val', [49, 49], [180, 1], file_time='')
slice.store_HDF5()