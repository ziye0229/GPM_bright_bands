import h5py
import numpy as np
import os
import math
import re
from tqdm import tqdm


class Oversampling:
    def __init__(self, s_folder_dir, d_folder_dir, slice_width, slice_num, file_time=''):
        self.root_dir = s_folder_dir
        self.des = d_folder_dir
        self.slice_width = slice_width
        self.slice_num = slice_num
        self.file_time = file_time

        self.files = self.get_file_list()
        self.oversampled_files_num = 0

    def __init_data__(self, factor_shape):
        self.generated_data = {'zFactor': np.empty([self.slice_num, self.slice_width, factor_shape[-2], factor_shape[-1]],
                                         dtype='float32'),
                     'zeroDeg': np.empty([self.slice_num, self.slice_width, factor_shape[-2]], dtype='int32'),
                     'flagBB': np.empty([self.slice_num, self.slice_width, factor_shape[-2]], dtype='int32'),
                     'BBTop': np.empty([self.slice_num, self.slice_width, factor_shape[-2]], dtype='int32'),
                     'BBBottom': np.empty([self.slice_num, self.slice_width, factor_shape[-2]], dtype='int32'),
                     'BBPeak': np.empty([self.slice_num, self.slice_width, factor_shape[-2]], dtype='int32')}
        self.data_len = 0

    def __init_index__(self, factor_shape):
        self.index_list = []
        step = math.ceil(factor_shape[0] / self.slice_num)
        for i in range(0, self.slice_num[0]):
            if i * step + self.slice_width <= factor_shape[0]:
                i_begin = i * step
                i_end = i * step + self.slice_width
            else:
                i_begin = factor_shape[0] - self.slice_width
                i_end = factor_shape[0] + 1
            self.index_list.append([i_begin, i_end])

    def oversample(self):
        for file in tqdm(self.files):
            self.oversample_file(file)

    def get_file_list(self):
        hdf5_files = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if os.path.splitext(file)[1] == '.HDF5' and re.match('2A.GPM.Ku.V9-20211125.' + self.file_time,
                                                                     os.path.splitext(file)[0]):
                    # print(os.path.splitext(file)[0])
                    hdf5_files.append(os.path.join(root, file))
        return hdf5_files

    def oversample_file(self, file_path):
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

            zFactorMeasured[zFactorMeasured < 12] = 0
            zFactorMeasured[zFactorMeasured >= 65] = 65
            zFactorMeasured = zFactorMeasured / 65

            ZeroDeg = np.array(ZeroDeg, dtype='int32')
            ZeroDeg[ZeroDeg == 177] = 176
            ZeroDeg -= 101
            ZeroDeg[ZeroDeg < 0] = 0

            flagBB = np.array(flagBB, dtype='int32')
            flagBB[flagBB > 0] = 2
            flagBB[flagBB == 0] = 1
            flagBB[flagBB < 0] = 0

            BBTop = np.array(BBTop, dtype='int32')
            BBTop -= 101
            BBTop[BBTop == -101] = 76
            BBTop[BBTop <= -1111] = 77
            BBTop[BBTop < 0] = 0
            BBBottom = np.array(BBBottom, dtype='int32')
            BBBottom -= 101
            BBBottom[BBBottom == -101] = 76
            BBBottom[BBBottom <= -1111] = 77
            BBBottom[BBBottom < 0] = 0
            BBPeak = np.array(BBPeak, dtype='int32')
            BBPeak -= 101
            BBPeak[BBPeak == -101] = 76
            BBPeak[BBPeak <= -1111] = 77
            BBPeak[BBPeak < 0] = 0

            for indexes in self.index_list:
                flag = flagBB[indexes[0]:indexes[1]][:]
                if flag[flag == 0].shape[0] < self.slice_width[0] * self.slice_width[1]:
                    self.generate_new_data(indexes, zFactorMeasured, ZeroDeg, flagBB, BBTop, BBBottom, BBPeak)

    def generate_new_data(self, indexes, zFactorMeasured, ZeroDeg, flagBB, BBTop, BBBottom, BBPeak):
        zFactor_sliced = np.flip(zFactorMeasured[indexes[0]:indexes[1]][:][:], axis=1)
        zeroDeg_sliced = np.flip(ZeroDeg[indexes[0]:indexes[1]][:], axis=1)
        flagBB_sliced = np.flip(flagBB[indexes[0]:indexes[1]][:], axis=1)
        BBTop_sliced = np.flip(BBTop[indexes[0]:indexes[1]][:], axis=1)
        BBBottom_sliced = np.flip(BBBottom[indexes[0]:indexes[1]][:], axis=1)
        BBPeak_sliced = np.flip(BBPeak[indexes[0]:indexes[1]][:], axis=1)
        self.add_data(zFactor_sliced, zeroDeg_sliced, flagBB_sliced, BBTop_sliced, BBBottom_sliced, BBPeak_sliced)

        for i in range(1, self.slice_width):
            new_indexes = [indexes[0]-1, indexes[1]-1]
            try:
                flag = flagBB[new_indexes[0]:new_indexes[1]][:]
                if flag[flag == 0].shape[0] < self.slice_width[0] * self.slice_width[1]:
                    self.generate_new_data_sub(new_indexes, zFactorMeasured, ZeroDeg, flagBB, BBTop, BBBottom, BBPeak)
            except:
                break
        for i in range(1, self.slice_width):
            new_indexes = [indexes[0]+1, indexes[1]+1]
            try:
                flag = flagBB[new_indexes[0]:new_indexes[1]][:]
                if flag[flag == 0].shape[0] < self.slice_width[0] * self.slice_width[1]:
                    self.generate_new_data_sub(new_indexes, zFactorMeasured, ZeroDeg, flagBB, BBTop, BBBottom, BBPeak)
            except:
                break

    def generate_new_data_sub(self, indexes, zFactorMeasured, ZeroDeg, flagBB, BBTop, BBBottom, BBPeak):
        zFactor_sliced = zFactorMeasured[indexes[0]:indexes[1]][:][:]
        zeroDeg_sliced = ZeroDeg[indexes[0]:indexes[1]][:]
        flagBB_sliced = flagBB[indexes[0]:indexes[1]][:]
        BBTop_sliced = BBTop[indexes[0]:indexes[1]][:]
        BBBottom_sliced = BBBottom[indexes[0]:indexes[1]][:]
        BBPeak_sliced = BBPeak[indexes[0]:indexes[1]][:]
        self.add_data(zFactor_sliced, zeroDeg_sliced, flagBB_sliced, BBTop_sliced, BBBottom_sliced,
                      BBPeak_sliced)
        zFactor_sliced = np.flip(zFactor_sliced, axis=1)
        zeroDeg_sliced = np.flip(zeroDeg_sliced, axis=1)
        flagBB_sliced = np.flip(flagBB_sliced, axis=1)
        BBTop_sliced = np.flip(BBTop_sliced, axis=1)
        BBBottom_sliced = np.flip(BBBottom_sliced, axis=1)
        BBPeak_sliced = np.flip(BBPeak_sliced, axis=1)
        self.add_data(zFactor_sliced, zeroDeg_sliced, flagBB_sliced, BBTop_sliced, BBBottom_sliced,
                      BBPeak_sliced)

    def add_data(self, zFactor_sliced, zeroDeg_sliced, flagBB_sliced, BBTop_sliced, BBBottom_sliced, BBPeak_sliced):
        if self.data_len == 0:
            self.__init_data__(zFactor_sliced.shape)
        self.generated_data['zFactor'][self.data_len] = zFactor_sliced
        self.generated_data['zeroDeg'][self.data_len] = zeroDeg_sliced
        self.generated_data['flagBB'][self.data_len] = flagBB_sliced
        self.generated_data['BBTop'][self.data_len] = BBTop_sliced
        self.generated_data['BBBottom'][self.data_len] = BBBottom_sliced
        self.generated_data['BBPeak'][self.data_len] = BBPeak_sliced
        self.data_len += 1
        if self.data_len == 7934:
            self.store_oversampled_HDF5()
            self.__init_data__(zFactor_sliced.shape)

    def store_oversampled_HDF5(self):
        self.oversampled_files_num += 1
        # with h5py.File(self.des + '/oversampled{}.HDF5'.format(self.oversampled_files_num), "w") as f:
        #     for key, value in self.generated_data.items():
        #         f.create_dataset(key, data=value)
        print(self.oversampled_files_num)


slice = Oversampling('./data/Ku/raw_val', './data/Ku/val', 49, 180, file_time='')
slice.oversample()
