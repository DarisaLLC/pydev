import sys
from pathlib import Path

import cv2
import numpy as np
import opencv_utils
from matplotlib import pyplot as plt
from skimage import draw
from skimage import img_as_float
from skimage import img_as_ubyte
from skimage import measure
import squares
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

bins = [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
        13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
        26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
        39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
        52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
        65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
        78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
        91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
       104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
       117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
       130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
       143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
       156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
       169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
       182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
       195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
       208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
       221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233,
       234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246,
       247, 248, 249, 250, 251, 252, 253, 254, 255]

def axis(x): return int((x + 16) // 32) % 8


def truePeaks(magnitudes, phases):
    tps = []
    axish = np.zeros(256, np.int32)
    ltps = [[], [], [], [], [], [], [], []]
    offsets = [[[0, -1], [0, 1]], [[-1, -1], [1, 1]], [[-1, 0], [1, 0]], [[-1, 1], [1, -1]]]

    height, width = magnitudes.shape[:2]
    for i in range(2, height - 2, 1):
        for j in range(2, width - 2, 1):
            angle = phases[i, j]
            mag = magnitudes[i, j]
            if mag < 100: continue
            ax = axis(angle)
            axish[angle] += 1
            m1 = m2 = mag
            if ax == 0 or ax == 4:  # left & right column
                m1 = magnitudes[i, j - 1]
                m2 = magnitudes[i, j + 1]
            elif ax == 1 or ax == 5:  # top left & bottom right
                m1 = magnitudes[i - 1, j - 1]
                m2 = magnitudes[i + 1, j + 1]
            elif ax == 2 or ax == 6:  # top & bottom row
                m1 = magnitudes[i - 1, j]
                m2 = magnitudes[i + 1, j]
            elif ax == 3 or ax == 7:  # bottom left and top right
                m1 = magnitudes[i - 1, j + 1]
                m2 = magnitudes[i + 1, j - 1]
            else:
                assert (False)

            if ((mag > m1 and mag >= m2) or (mag >= m1 and mag > m2)):
                ltps[ax].append([j, i, ax, angle, mag])

    tps = np.concatenate(ltps)

    return (tps, axish)



def process(img, half_size=1):
    hs = half_size - int(half_size) % 1
    block_size = 2 * hs + 1

    sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=block_size)
    sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=block_size)
    phases = cv2.phase(sobelx, sobely, False)
    #    phases = np.unwrap(phases, np.pi + np.pi) / (2 * np.pi)
    phases = np.divide(phases, np.pi + np.pi)
    phases = img_as_ubyte(phases)
    magnitudes = np.sqrt(sobelx ** 2 + sobely ** 2)

    return (magnitudes, phases)


if __name__ == '__main__':
    result = None
    display = None
    tcontours = None
    contours = None


    def data_circle():
        arr = np.zeros((200, 200))
        rr, cc = draw.circle(100, 100, 67)
        arr[rr, cc] = 200
        img = cv2.GaussianBlur(arr, (7, 7), sigmaX=1.2)
        return img


    contours = None
    corners = None

    if len(sys.argv) < 2:
        # Construct some test data
        r = img_as_float(data_circle())
        result = process(np.uint8(r), 1)
        contours = measure.find_contours(r, 0.5)
        display = r
    else:
        if not Path(sys.argv[1]).is_file() or not Path(sys.argv[1]).exists():
            print(sys.argv[1] + '  Does not exist ')

        lab_tuple = opencv_utils.load_reduce_convert(sys.argv[1], 2)
        corners = squares.find_squares(lab_tuple[0])
        display = opencv_utils.convert_lab2rgb(lab_tuple)
        result = process(lab_tuple[0], 1)

    tps, axh = truePeaks(result[0], result[1])
    truepeaks = np.array(tps, dtype='f')


    fig = plt.figure(figsize=(22, 15))



    ax2 = plt.subplot2grid((3, 3), (0, 0))
    ax2.imshow(result[0], cmap='gray')
    ax2.set_title('Gradient Magnitude')
    ax2.plot(corners[:, 0], corners[:, 1], '+g', markersize=3)
    ax2.set_title('Image + CORNERS ')

    ax2 = plt.subplot2grid((3, 3), (0, 1))
    ax2.imshow(result[1], cmap='gray')
    ax2.set_title('Gradient Angle')

    ax1 = plt.subplot2grid((3, 3), (1, 0))
    ax1.imshow(display, cmap='gray')
    ax1.plot(truepeaks[:, 0], truepeaks[:, 1], '+r', markersize=3)
    ax1.set_title('Image + TP ')

    # Display histogram
    ax_hist = plt.subplot2grid((3, 3), (1, 1))
    ax_hist.hist(result[1].ravel(), bins=256,density=True)
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel(' Direction / 2 PI')
  #  ax_hist.set_xlim(0, 1.0)
   # ax_hist.set_yticks([])

    plt.show()
