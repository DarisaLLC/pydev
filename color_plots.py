# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import coloralgo
import cv2
import numpy as np
import sobelplus
from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter
from skimage import color
from skimage import exposure
from skimage import img_as_float
from skimage import io
from skimage.io import imsave
from coloralgo import get_dominant_color, omf


# @memorize.Memorize
def fetch_image_for_image_file(filename):
    return io.imread(filename)


def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()
    ax_img.set_adjustable('box-forced')

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


def plot2dMutual(x, y):
    nullfmt = NullFormatter()  # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max ( np.max ( np.abs ( x ) ), np.max ( np.abs ( y ) ) )
    xymin = min ( np.min ( np.abs ( x ) ), np.min ( np.abs ( y ) ) )
    lim = (int ( xymax / binwidth ) + 1) * binwidth
    llim = (int ( xymin / binwidth ) + 1) * binwidth

    axScatter.set_xlim ( (llim, lim) )
    axScatter.set_ylim ( (llim, lim) )

    bins = np.arange ( llim - binwidth, lim + binwidth, binwidth )
    axHistx.hist(x, bins=bins)
    axHisty.hist(y, bins=bins, orientation='horizontal')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    plt.show()


def main_lab(img):
    original = img.copy()
    wbm = coloralgo.grey_world_median(img)
    wb = coloralgo.gray_world(img)

    f, axs = plt.subplots(2, 3, figsize=(20, 10), frameon=False,
                          subplot_kw={'xticks': [], 'yticks': []})
    axs[0, 0].imshow(original)
    axs[0, 1].imshow(wb)
    axs[0, 2].imshow(wbm)

    (x, y, c) = coloralgo.LabJointHistogram(original)
    axs[1, 0].scatter(x, y, marker='.', cmap='PuBu_r')
    axs[1, 0].set_title('Original')
    axs[1, 0].set_xlabel('b')
    axs[1, 0].set_ylabel('a')

    (x, y, c) = coloralgo.LabJointHistogram(wb)
    axs[1, 1].scatter(x, y)
    axs[1, 1].set_title('GW White Balance')
    axs[1, 1].set_xlabel('b')
    axs[1, 1].set_ylabel('a')

    (x, y, c) = coloralgo.LabJointHistogram(wbm)
    axs[1, 2].scatter(x, y)
    axs[1, 2].set_title('Modified GW White Balance')
    axs[1, 2].set_xlabel('b')
    axs[1, 2].set_ylabel('a')

    plt.autoscale
    plt.show()


def main_lab_collection(images):
    count = len(images)

    f, axs = plt.subplots(count, 2, figsize=(20, 10), frameon=False,
                          subplot_kw={'xticks': [], 'yticks': []})


    pos = 0
    for image in images:
        hwc = image.shape
        if (hwc[2] == 3):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        res = sobelplus.process(gray, 5)
        axs[pos, 0].imshow(res[1], cmap=plt.cm.gray)

        towhite = coloralgo.DistanceToWhite(image)
        score = str(int(towhite['score'] * 255 * 255))
        axs[pos, 1].imshow(towhite['image'], cmap=plt.cm.gray)
        axs[pos, 1].annotate(score, xy=(200, 100), xytext=(1000, 1500), color='red')
        pos += 1

    plt.autoscale
    plt.show()


def main_hue(img):
    rotred = coloralgo.rotate_hue(img, -10)
    imsave('/Users/arman/Desktop/240_32_10.jpg', rotred)
    f, axs = plt.subplots(3, 2, figsize=(20, 10), frameon=False,
                          subplot_kw={'xticks': [], 'yticks': []})
    axs[0, 0].imshow(img)
    axs[0, 1].imshow(rotred)

    imghue = coloralgo.get_hue(img)
    plot_img_and_hist(imghue, axs[1, :])

    rotredhue = coloralgo.get_hue(rotred)
    plot_img_and_hist(rotredhue, axs[2, :])

    plt.autoscale
    plt.show()


def main_speed_cv_lab(img):
    import time
    import cv2
    from skimage import data, color

    image = data.chelsea()

    t = time.time()
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    t_cv = time.time() - t

    t = time.time()
    lab = color.rgb2lab(image)
    t_sk = time.time() - t

    print('cv:', t_cv)
    print('skimage:', t_sk)
    print('factor:', t_sk / t_cv)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        exit(1)
    print(sys.argv[1])
    if Path(sys.argv[1]).is_file():
        img = fetch_image_for_image_file(sys.argv[1])
        if img.ndim > 3:
            rgb = color.rgba2rgb(img)
        elif img.ndim == 3:
            rgb = img
        else:
            print('Not a color image')
            exit(1)

        omfs = omf(img)
        dc = get_dominant_color(img)
        print(dc)

        main_lab(rgb)
        # main_speed_cv_lab(rgb)

    elif Path(sys.argv[1]).is_dir():
        coll = io.ImageCollection(sys.argv[1] + '/*.jpg')
        print(len(coll))
        main_lab_collection(coll)
