import h5py
import numpy as np


f = h5py.File('2A.GPM.DPR.V8-20180723.20210101-S013133-E030407.038882.V06A.HDF5', 'r')
CSF = f['NS']['CSF']
BBTop = CSF['binBBTop']
BBBottom = CSF['binBBBottom']
BBPeak = CSF['binBBPeak']
VER = f['NS']['VER']
ZeroDeg = VER['binZeroDeg']
data = [['X', 'Y', 'BBTop', 'BBBottom', 'BBPeak', 'ZeroDeg']]

for pixels_BBTop, pixels_BBBottom, pixels_BBPeak, pixels_ZeroDeg, i in zip(BBTop, BBBottom, BBPeak, ZeroDeg, range(0, BBTop.shape[0])):
    for pixel_BBTop, pixel_BBBottom, pixel_BBPeak, pixel_ZeroDeg, j in zip(pixels_BBTop, pixels_BBBottom, pixels_BBPeak, pixels_ZeroDeg, range(0, BBTop.shape[1])):
        if (-1111 not in [int(pixel_BBTop), int(pixel_BBBottom), int(pixel_BBPeak)]) or ([int(pixel_BBTop), int(pixel_BBBottom), int(pixel_BBPeak)] == [-1111, -1111, -1111]):
            continue
        else:
            print(pixel_BBTop, pixel_BBBottom, pixel_BBPeak, pixel_ZeroDeg)
        # if pixel_BBTop == -1111:
        #     pixel_BBTop = '-'
        # else:
        #     pixel_BBTop = int(pixel_BBTop)
        # if pixel_BBBottom == -1111:
        #     pixel_BBBottom = '-'
        # else:
        #     pixel_BBBottom = int(pixel_BBBottom)
        # if pixel_BBPeak == -1111:
        #     pixel_BBPeak = '-'
        # else:
        #     pixel_BBPeak = int(pixel_BBPeak)
        #     data.append([i, j, pixel_BBTop, pixel_BBBottom, pixel_BBPeak, int(pixel_ZeroDeg)])
f.close()