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
import math

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




def gradient_magnitude_mask(img, sobel_kernel=3, threshold=(0, 255)):
    """
    Masks the image based on gradient magnitude.

    Parameters
    ----------
    image           : Image to mask.
    sobel_kernel    : Kernel of the Sobel gradient operation.
    threshold       : Magnitude threshold for it to make it to appear in the mask.

    Returns
    -------
    Image mask with 1s in activations and 0 in other pixels.
    """
    # Take the gradient in x and y separately
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the magnitude
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    magnitude = (magnitude * 255 / np.max(magnitude)).astype(np.uint8)
    # Create a binary mask where mag thresholds are met
    mask = np.zeros_like(magnitude)
    mask[(magnitude >= threshold[0]) & (magnitude <= threshold[1])] = 1
    # Return this mask as your binary_output image
    return mask




def truePeaks(magnitudes, phases_ubyte, phases, threshold):
    height, width = magnitudes.shape[:2]
    if threshold == 0:
        threshold = np.median(magnitudes.flatten())
    tps = []
    u = np.zeros((height, width))
    v = np.zeros((height, width))
    axish = np.zeros(256, np.int32)
    ltps = [[], [], [], [], [], [], [], []]
    offsets = [[[0, -1], [0, 1]], [[-1, -1], [1, 1]], [[-1, 0], [1, 0]], [[-1, 1], [1, -1]]]
    sUU = 0.0
    sVV = 0.0
    sUV = 0.0
    sUR = 0.0
    sVR = 0.0

    for i in range(2, height - 2, 1):
        for j in range(2, width - 2, 1):
            angle = phases_ubyte[i, j]
            theta = phases[i, j]
            mag = magnitudes[i, j]
            if mag < threshold: continue
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

            if (((mag >= m1 and mag > m2)) and (m1 != m2)):
                ltps[ax].append([j, i, angle, mag, m1, m2])
                uij = math.sin(theta + math.pi / 2)
                vij = math.cos(theta + math.pi / 2)
                u[i, j] = uij
                v[i, j] = vij
                r = j * uij + i * vij
                sUU += uij * uij
                sVV += vij * vij
                sUV += uij + vij
                sUR += uij * r
                sVR += vij * r

    # determinant
    detA = sUU * sVV - sUV * sUV
    moc = None
    if detA > 0:
        moc = np.zeros(2)
        moc[0] = (sUR * sVV - sUV * sVR) / detA
        moc[1] = (sVR * sUU - sUV * sUR) / detA

    tps = np.concatenate(ltps)

    return (tps, u, v, axish, moc, threshold)


def sobel_detect(img, half_size=1):
    hs = half_size - int(half_size) % 1
    block_size = 2 * hs + 1

    sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=block_size)
    sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=block_size)
    phases = cv2.phase(sobelx, sobely, False)
    #    phases = np.unwrap(phases, np.pi + np.pi) / (2 * np.pi)
    phases = np.divide(phases, np.pi + np.pi)
    phases_ubyte = img_as_ubyte(phases)
  
    magnitudes = np.sqrt(sobelx ** 2 + sobely ** 2)
    return (magnitudes, phases_ubyte, phases)


def gradient_all(img, threshold=10, half_size=1):
    result = sobel_detect(img, half_size)
    output_tuple = truePeaks(result[0], result[1], result[2], half_size)
    return output_tuple

if __name__ == '__main__':
    result = None
    display = None
    tcontours = None
    contours = None
    synth_center_col = 7
    synth_center_row = 8
    synth_case = False
    gdm = None
    def data_circle():
        arr = np.zeros((16, 16), dtype=np.uint8)
        arr = arr + 200
        rr, cc = draw.circle(synth_center_row, synth_center_col, 5)
        arr[rr, cc] = 0
        img = cv2.GaussianBlur(arr, (7, 7), sigmaX=1.2)
        return img

    def data_corner():
        arr = np.zeros((16, 16), dtype=np.uint8)
        arr = arr + 200
        arr[0:7, 0:7] = 0
        img = cv2.GaussianBlur(arr, (7, 7), sigmaX=1.2)
        return img

    if len(sys.argv) < 2:
        # Construct some test data
        r = data_corner()
        result = sobel_detect(r, 1)
        display = img_as_float(r)
        gdm = gradient_magnitude_mask(r, (100, 255))
        synth_case = True
    else:
        if not Path(sys.argv[1]).is_file() or not Path(sys.argv[1]).exists():
            print(sys.argv[1] + '  Does not exist ')
        lab_tuple = opencv_utils.load_reduce_convert(sys.argv[1], 3)
        display = opencv_utils.convert_lab2rgb(lab_tuple)
        img = cv2.GaussianBlur(lab_tuple[0], (7,7), sigmaX=3.0)
        result = sobel_detect(img, 1)
        gdm = gradient_magnitude_mask(img, 3, (10, 255))
        gdm = cv2.bitwise_and(result[1], result[1], mask=gdm)
        cv2.imwrite("/Users/arman/tmp/directionMask.png", gdm)
        cv2.imwrite("/Users/arman/tmp/blur.png", img)


    dims = display.shape
    height = dims[0]
    width = dims[1]
    tps, u, v, axh, moc, threshold = truePeaks(result[0], result[1], result[2], 100)

    if synth_case:
        dx = moc[0] - synth_center_col
        dy = moc[1] - synth_center_row
        print((dx, dy))
    else: print(moc)

    truepeaks = np.array(tps, dtype='f')
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)

    fig = plt.figure(figsize=(22, 15))

    ax2 = plt.subplot2grid((3, 3), (0, 0))
    ax2.imshow(display, cmap='gray')
    ax2.set_title('Input Image')

    ax2 = plt.subplot2grid((3, 3), (0, 1))
    ax2.imshow(result[2], cmap='gray')
    ax2.set_title('Gradient Angle')

    ax2 = plt.subplot2grid((3, 3), (0, 2))
    ax2.imshow(gdm, cmap='gray')
    ax2.set_title('Masked Gradient Angle')

    skip = (slice(None, None, 3), slice(None, None, 3))
    ax2 = plt.subplot2grid((3, 3), (1, 0))
    ax2.imshow(display, cmap='gray')
    ax2.quiver(u[::,3],v[::,3], units='x', scale=1, pivot='mid', color='r')
    ax2.set_title('Image + TP ')
    ax2.xaxis.set_ticks([])
    ax2.yaxis.set_ticks([])
    ax2.set_aspect('equal')

    # Display histogram
    ax_hist = plt.subplot2grid((3, 3), (1, 1))
    ax_hist.hist(axh, bins=256, range=(0, 255))
    ax_hist.set_xlabel(' Direction Histogram')

    ax2 = plt.subplot2grid((3, 3), (1, 2))
    ax2.imshow(result[0], cmap='gray')
    ax2.set_title('Gradient Magnitude')

    plt.show()
