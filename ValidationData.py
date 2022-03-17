import h5py
import numpy as np
import os
import random
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt


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
    flagBB = CSF['flagBB']
    BBTop = CSF['binBBTop']
    BBBottom = CSF['binBBBottom']
    BBPeak = CSF['binBBPeak']
    VER = f['NS']['VER']
    ZeroDeg = VER['binZeroDeg']

    for pixels_flagBB, pixels_BBTop, pixels_BBBottom, pixels_BBPeak, pixels_ZeroDeg, i in zip(flagBB, BBTop, BBBottom, BBPeak, ZeroDeg,
                                                                               range(0, BBTop.shape[0])):
        for pixel_flagBB, pixel_BBTop, pixel_BBBottom, pixel_BBPeak, pixel_ZeroDeg, j in zip(pixels_flagBB, pixels_BBTop, pixels_BBBottom,
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

            if ((-1111 not in [int(pixel_BBTop), int(pixel_BBBottom), int(pixel_BBPeak)]) and int(pixel_flagBB) < 0) or \
                    (([int(pixel_BBTop), int(pixel_BBBottom), int(pixel_BBPeak)] == [-1111, -1111, -1111]) and int(pixel_flagBB) >= 0):
                print(pixel_flagBB, pixel_BBTop, pixel_BBBottom, pixel_BBPeak, pixel_ZeroDeg)


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
    """
    探究亮带峰值与零度层之间距离的分布规律
    结论：呈均值为-2的正态分布
    :param f:
    :return:
    """
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
                if -15 > (int(pixel_ZeroDeg) - int(pixel_BBPeak)) or (int(pixel_ZeroDeg) - int(pixel_BBPeak)) > 12:
                    print(int(pixel_ZeroDeg), int(pixel_BBPeak))
    return Counter(distance)


def test4(f):
    """
    探究无降雨、有降雨无亮带、有降雨有亮带之间的比例关系
    :param f:
    :return:
    """
    CSF = f['NS']['CSF']
    flagBB = CSF['flagBB']
    flagBB = np.array(flagBB, dtype='int32')
    flagBB_neg = flagBB[flagBB < 0]
    flagBB_0 = flagBB[flagBB == 0]
    flagBB_pos = flagBB[flagBB > 0]

    return flagBB.shape[0]*flagBB.shape[1], [flagBB_neg.shape[0], flagBB_0.shape[0], flagBB_pos.shape[0]]


def test5(f):
    """
    探究有亮带和检测为层云降水之间的关系
    结论：观测到两代是层云降雨的充分条件，但不是必要条件
    :param f:
    :return:
    """
    CSF = f['NS']['CSF']
    flagBB = CSF['flagBB']
    typePrecip = CSF['typePrecip']
    flagBB = np.array(flagBB, dtype='int32')
    typePrecip = np.array(typePrecip, dtype='int32')

    for pixcels_flagBB, pixcels_typePrecip in zip(flagBB, typePrecip):
        for pixcel_flagBB, pixcel_typePrecip in zip(pixcels_flagBB, pixcels_typePrecip):
            if pixcel_flagBB <= 0 and (pixcel_typePrecip // 10000000) == 1:
                if pixcel_flagBB != 0:
                    print(pixcel_flagBB, pixcel_typePrecip)


def test6(f):
    """
    探究亮带顶位置的分布情况
    结论：两组正态分布叠加
    :param f:
    :return:
    """

    top_position = []
    CSF = f['NS']['CSF']
    BBTop = CSF['binBBTop']
    for pixcels_top in BBTop:
        for top in pixcels_top:
            if int(top) > 0:
                top_position.append(int(top))
    return Counter(top_position)


def test7(f):
    """
    探究零度层的分布情况
    结论：最小值出现在123层
    :param f:
    :return:
    """
    zero_position = []
    VER = f['NS']['VER']
    ZeroDeg = VER['binZeroDeg']
    for pixcels_zero in ZeroDeg:
        for zero in pixcels_zero:
            if int(zero) > 0:
                zero_position.append(int(zero))
    return Counter(zero_position)

def test8(f):
    """
    探究校正后的雷达回波的数据范围
    :param f:
    :return:
    """
    zfactor = f['NS']['SLV']['zFactorCorrected']
    zfactor = np.array(zfactor, dtype='float32')
    zfactor = zfactor[zfactor >= -200]
    zfactor_min = zfactor.min()
    zfactor_max = zfactor.max()
    return zfactor_min, zfactor_max


cnt = Counter()
total_shape_sum = 0
shape_list_sum = [0, 0, 0]
zfactor_min = np.inf
zfactor_max = -np.inf
files = getFileList('F:/2ADPR')
plt.rcParams['font.sans-serif'] = ['SimHei']    #显示中文标签
plt.rcParams['axes.unicode_minus'] = False      #这两行需要手动设置
# print(files)
files = random.sample(files, 400)
for file_path in files:
    f = h5py.File(file_path, 'r')
    # test1(f)
    # test2(f)
    # test3(f)
    # cnt += test3(f)

    # total_shape, shape_list = test4(f)
    # total_shape_sum += total_shape
    # for i in range(0, len(shape_list_sum)):
    #     shape_list_sum[i] += shape_list[i]

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

    # test5(f)

    # cnt += test7(f)

    min_num, max_num = test8(f)
    print(min_num, max_num)
    if zfactor_max < max_num:
        zfactor_max = max_num
    if zfactor_min > min_num:
        zfactor_min = min_num

    f.close()

# # 配合test3()显示零度层与亮带距离分布柱状图
# x = []
# y = []
# rst = sorted(cnt.items(), key=lambda x: x[0], reverse=False)
# for item in rst:
#     x.append(item[0])
#     y.append(item[1])
# fig, ax = plt.subplots(figsize=(10, 7))
# ax.bar(x=x, height=y)
# ax.set_title("零度层与亮带距离分布", fontsize=15)
# print(cnt)
# for i in range(0, len(shape_list_sum)):
#     print('{}%'.format(shape_list_sum[i]/total_shape_sum*100))

# 配合test6()\test7()显示分布柱状图
# x = []
# y = []
# rst = sorted(cnt.items(), key=lambda x: x[0], reverse=False)
# for item in rst:
#     x.append(item[0])
#     y.append(item[1])
# fig, ax = plt.subplots(figsize=(10, 7))
# ax.bar(x=x, height=y)
# ax.set_title("零度层位置分布", fontsize=15)
# print(cnt)
# print(min(x))
# plt.show()

print(zfactor_min, zfactor_max)


# i = 24
# o = 24
# for k in range(2, 7):
#     for p in range(0, 4):
#         for s in range(1, 4):
#             # if (o + 2 * p - k) % s == 0:
#             #     if o == s * (i - 1) - 2 * p + k:
#             #         print('k:{}, p:{}, s:{}'.format(k, p, s))
#             # else:
#             #     if o == s * (i - 1) - 2 * p + k + (o + 2 * p - k) % s:
#             #         print('k:{}, p:{}, s:{}'.format(k, p, s))
#
#             if o == (i - k + 2 * p) // s + 1:
#                 print('k:{}, p:{}, s:{}'.format(k, p, s))
