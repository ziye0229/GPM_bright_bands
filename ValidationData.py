import h5py
import numpy as np
import os
import random
import tqdm
from collections import Counter


def getFileList(root_dir):
    hdf5_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.HDF5':
                hdf5_files.append(os.path.join(root, file))
    return hdf5_files


def test1(f):
    """
    查看BBTop, BBBottom, BBPeak, ZeroDeg数据完整度
    结论：
    BBTop, BBBottom, BBPeak不会出现部分缺失的情况，但会出现全为“0”或全为“-1111”的情况
    ZeroDeg不会缺失
    :param f:
    :return:
    """
    CSF = f['NS']['CSF']
    BBTop = CSF['binBBTop']
    BBBottom = CSF['binBBBottom']
    BBPeak = CSF['binBBPeak']
    VER = f['NS']['VER']
    ZeroDeg = VER['binZeroDeg']

    for pixels_BBTop, pixels_BBBottom, pixels_BBPeak, pixels_ZeroDeg, i in zip(BBTop, BBBottom, BBPeak, ZeroDeg,
                                                                               range(0, BBTop.shape[0])):
        for pixel_BBTop, pixel_BBBottom, pixel_BBPeak, pixel_ZeroDeg, j in zip(pixels_BBTop, pixels_BBBottom,
                                                                               pixels_BBPeak, pixels_ZeroDeg,
                                                                               range(0, BBTop.shape[1])):
            # 0度层数据是否完整
            if 0 <= int(pixel_ZeroDeg) <= 177:
                # BBTop, BBBottom, BBPeak数据是否部分缺失
                if (-1111 not in [int(pixel_BBTop), int(pixel_BBBottom), int(pixel_BBPeak)]) or \
                        ([int(pixel_BBTop), int(pixel_BBBottom), int(pixel_BBPeak)] == [-1111, -1111, -1111]):
                    continue
                elif (0 not in [int(pixel_BBTop), int(pixel_BBBottom), int(pixel_BBPeak)]) or \
                        ([int(pixel_BBTop), int(pixel_BBBottom), int(pixel_BBPeak)] == [0, 0, 0]):
                    continue
                else:
                    print(pixel_BBTop, pixel_BBBottom, pixel_BBPeak, pixel_ZeroDeg)
            else:
                print(pixel_BBTop, pixel_BBBottom, pixel_BBPeak, pixel_ZeroDeg)


def test2(f):
    """
    探究当地表温度<=0℃(binZeroDeg>=176)时，是否还会出现亮带
    结论：
    还会有
    :param f:
    :return:
    """
    CSF = f['NS']['CSF']
    flagBB = CSF['flagBB']
    # BBTop = CSF['binBBTop']
    # BBBottom = CSF['binBBBottom']
    # BBPeak = CSF['binBBPeak']
    VER = f['NS']['VER']
    ZeroDeg = VER['binZeroDeg']
    print(flagBB)
    np_flagBB = np.array(flagBB)
    print(flagBB)

    for pixels_flagBB, pixels_ZeroDeg in zip(flagBB, ZeroDeg):
        for pixel_flagBB, pixel_ZeroDeg in zip(pixels_flagBB, pixels_ZeroDeg):
            if int(pixel_ZeroDeg) >= 176 and int(pixel_flagBB) > 0:
                print(pixel_flagBB, pixel_ZeroDeg)


def test3(f):
    distance = []
    CSF = f['NS']['CSF']
    flagBB = CSF['flagBB']
    # BBTop = CSF['binBBTop']
    # BBBottom = CSF['binBBBottom']
    BBPeak = CSF['binBBPeak']
    VER = f['NS']['VER']
    ZeroDeg = VER['binZeroDeg']

    for pixels_flagBB, pixels_BBPeak, pixels_ZeroDeg in zip(flagBB, BBPeak, ZeroDeg):
        for pixel_flagBB, pixel_BBPeak, pixel_ZeroDeg in zip(pixels_flagBB, pixels_BBPeak, pixels_ZeroDeg):
            if int(pixel_flagBB) > 0:
                distance.append(int(pixel_ZeroDeg) - int(pixel_BBPeak))

    return Counter(distance)






cnt = Counter()
files = getFileList('data/train')
# print(files)
files = random.sample(files, 100)
for file_path in tqdm(files, desc='Processing'):
    f = h5py.File(file_path, 'r')
    # test1(f)
    # test2(f)
    cnt += test3(f)
    print(cnt)
    # print('Finish!\t', file_path)

    # CSF = f['NS']['CSF']
    # flagBB = CSF['flagBB']
    # PRE = f['NS']['PRE']
    # zFactorMeasured = PRE['zFactorMeasured']
    # # print(flagBB.shape)
    # # np_flagBB = np.array(flagBB, dtype='int32')
    # # print(np_flagBB)
    # np_factor = np.array(zFactorMeasured, dtype='float32')
    # np_factor[np_factor <= -28888.0] = 0
    # print(np_factor.min(), np_factor.max())
    
    f.close()

# i = 176
# o = 88
# for k in range(2, 7):
#     for p in range(0, 4):
#         for s in range(1, 4):
#             # if (o + 2 * p - k) % s == 0:
#             #     if o == s * (i - 1) - 2 * p + k:
#             #         print('k:{}, p:{}, s:{}'.format(k, p, s))
#             # else:
#             #     if o == s * (i - 1) - 2 * p + k + (o + 2 * p - k) % s:
#             #         print('k:{}, p:{}, s:{}'.format(k, p, s))

#             if o == (i - k + 2 * p) // s + 1:
#                 print('k:{}, p:{}, s:{}'.format(k, p, s))