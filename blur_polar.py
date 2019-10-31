import cv2
import numpy as np
from urllib.request import urlopen
from matplotlib import pyplot as plt
import math
from zscore_1d import find_inflections
import matplotlib.patches as patches
from pathlib import Path

def get_channels(image):
    a, b, c = cv2.split(image)
    return (a, b, c)


def similarity(image_a, image_b):
    assert (image_a.shape == image_b.shape)
    dims = image_a.shape

    def channel_similarity(chan_a, chan_b):
        res = cv2.matchTemplate(chan_a, chan_b, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        return max_val * max_val

    if dims[2] is 1:
        return channel_similarity(image_a, image_b)
    else:
        a1, a2, a3 = get_channels(image_a)
        b1, b2, b3 = get_channels(image_b)
        r1 = channel_similarity(a1, b1)
        r2 = channel_similarity(a2, b2)
        r3 = channel_similarity(a3, b3)
        return math.pow(r1 * r2 * r3, 1 / 3.0)




def path_to_image(dpath):
    image_Gra = cv2.imread(dpath, cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
    image_BGR = cv2.imread(dpath, cv2.IMREAD_COLOR)
    return image_Gra, image_BGR



def linear_2_Polar(img, xm, ym):
    value = np.sqrt(((img.shape[0] / 1.0) ** 2) + ((img.shape[1] / 1.0) ** 2))
    polar_image = cv2.linearPolar(img, (xm, ym), value, cv2.WARP_FILL_OUTLIERS)
    polar_image = polar_image.astype(np.uint8)
    return (polar_image)


def polar_2_Linear(img, polar_img, xm, ym):
    value = np.sqrt(((img.shape[0] / 1.0) ** 2) + ((img.shape[1] / 1.0) ** 2))
    polar_rad = cv2.linearPolar(polar_img, (xm, ym), value, cv2.WARP_INVERSE_MAP)
    return (polar_rad)

#path = './projects/pupil/images/'
def blur_Radial(source, size, xm, ym, low_high = (-5,5), path = None, plot = None):
    filter_V = np.zeros((size, size))
    filter_V[:, int((size - 1) / 2)] = np.ones(size)
    filter_V = filter_V / size

    high_ = low_high[1]
    low_ = low_high[0]
    sm_dims = high_ - low_ + 1
    sm = np.zeros((sm_dims, sm_dims))
    image_RGB = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

    sum = sumsq = None
    cnt = 0
    for dx in range(low_, high_ + 1):
        for dy in range(low_, high_ + 1):
            polar_image = linear_2_Polar(source, xm + dx, ym + dy)
            polar_rad = cv2.filter2D(polar_image, -1, filter_V)
            if sum is None:
                sum = np.zeros(polar_rad.shape, np.float32)
            if sumsq is None:
                sumsq = np.zeros(polar_rad.shape, np.float32)
            dx2 = cv2.accumulate(polar_rad, sum)
            dy2 = cv2.accumulateSquare(polar_rad, sumsq)
            cnt = cnt + 1

            xx = dx + high_
            yy = dy + high_

            if not (path is None) and Path(path).exists():
                fname = path + 'polar_rad_%s_%s.png' % (xx, yy)
                cv2.imwrite(fname, polar_rad)

            polar_rad = polar_2_Linear(source, polar_rad, xm + dx, ym + dy)
            if dx == 0 and dy == 0:
                polar_rad_0_0 = polar_rad
            sm[dx + high_, dy + high_] = similarity(source, polar_rad)

    sumsq = cv2.multiply(sumsq, cnt)
    var_image = cv2.multiply(sum, sum)
    var_image = cv2.subtract(sumsq, var_image)
    var_image = cv2.divide(var_image, cnt * (cnt - 1))
    c1, c2, c3 = cv2.split(var_image)
    cc = (c1 + c2 + c3) / 3
    ydata = cv2.reduce(cc, 0, cv2.REDUCE_SUM, dtype=-1).flatten()
    idx = 0
    for yd in ydata:
        print('%d %d' % (idx, yd))
        idx = idx + 1

    xdata = np.arange(len(ydata))
    resd = find_inflections(xdata, ydata)
    result = resd['inflection']
    first_valley = resd['xvalleyes']
    pupil_radii = (first_valley[0] + first_valley[1]) / 2.0

    if plot:
        f1, ax1 = plt.subplots(1)
        ax1.set_aspect('equal')
        ax1.imshow(image_RGB)
        ax1.set_title("Pupil Results")
        c = patches.Circle((xm, ym), pupil_radii, color='red', linewidth=2, fill=False)
        ax1.add_patch(c)
        plt.show()

    print(resd)



def _main():
    print('Select Image:')
    print('a - frame0000')
    print('b - frame0003')
    print('c - frame0004')
    print('d - frame0005')
    print('e - frame0006')
    print('f - frame0007')
    print('h - polar_test')
    image_selected = str(input())

    color_selected = str('b')
    blur_selected = str('d')
    size = 50

    print('Select Coordinates: (1 / 512)')
    xm = int(input('X: ')) - 1
    ym = int(input('Y: ')) - 1

    if image_selected == 'a':
        lbl_Source = 'frame0000'
        url = './projects/pupil/images/frame0000.jpg'
    elif image_selected == 'b':
        lbl_Source = 'frame0003'
        url = './projects/pupil/images/frame0003.jpg'
    elif image_selected == 'c':
        lbl_Source = 'frame0000'
        url = './projects/pupil/images/frame0004.jpg'
    if image_selected == 'd':
        lbl_Source = 'frame0000'
        url = './projects/pupil/images/frame0005.jpg'
    elif image_selected == 'e':
        lbl_Source = 'frame0003'
        url = './projects/pupil/images/frame0006.jpg'
    elif image_selected == 'f':
        lbl_Source = 'frame0000'
        url = './projects/pupil/images/frame0007.jpg'
    if image_selected == 'g':
        lbl_Source = 'frame0000'
        url = './projects/pupil/images/frame0008.jpg'
    elif image_selected == 'h':
        lbl_Source = 'polar_test'
        url = './projects/pupil/images/polar_test.png'

    source, source_Color = path_to_image(url)
    source_Selected = source_Color

    Image_Output_Blur = blur_Radial(source_Selected, size, xm, ym, low_high = (-5,5), path = None, plot = True)

def main():
    while True:
        _main()
        print('')
        One_More_Time = str(input('Another Filter? (Y/N): ')).lower()
        if One_More_Time == 'y':
            continue
        else:
            break


main()

