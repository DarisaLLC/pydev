import sys
import cv2 as cv

import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from scipy.signal import find_peaks


def image_projection(img):
    """
    """
    # change to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Change to numpy array format
    nb = np.array(gray)

    x_sum = cv2.reduce(gray, 0, cv2.REDUCE_SUM, dtype=cv2.CV_64F)
    y_sum = cv2.reduce(gray, 1, cv2.REDUCE_SUM, dtype=cv2.CV_64F)

    # get height and weight
    x = gray.shape[1]
    y = gray.shape[0]

    # division the result by height and weight
    x_sum = x_sum / y
    y_sum = y_sum / x

    x_sum = x_sum.transpose()

    x_arr = np.arange(x)
    y_arr = np.arange(y)

    xn = np.array(x_sum)
    yn = np.array(y_sum)
    x_peaks, _ = find_peaks(xn.ravel(), height=0, width=10)
    y_peaks, _ = find_peaks(yn.ravel(), height=0, width=10)

    return x_arr, xn, x_peaks, y_arr, yn, y_peaks




import sys

from math_utils import medianFilter

if __name__ == '__main__':

    name = sys.argv[1]
    img_color = cv2.imread(name)
    img_color = img_color[255:1223,511:1223]

    rows, cols, cns = img_color.shape
    f, axs = plt.subplots(3, 2, figsize=(20, 10), frameon=False,
                          subplot_kw={'xticks': [], 'yticks': []})
    x_arr, xn, x_peaks, y_arr, yn, y_peaks = image_projection(img_color)

    med_xn = medianFilter(xn.ravel())
    med_yn = medianFilter(yn.ravel())

    axs[0, 0].plot(x_arr, xn)
    axs[0, 0].plot(x_peaks, xn[x_peaks], "x")
    axs[0, 0].grid(True)
    axs[0, 1].plot(y_arr, yn)
    axs[0, 1].plot(y_peaks, yn[y_peaks], "x")
    axs[0, 1].grid(True)

    axs[1, 0].plot(x_arr, med_xn)
    axs[1, 0].grid(True)
    axs[1, 1].plot(y_arr, med_yn)
    axs[1, 1].grid(True)

    plt.show()

    # display the image with projection peaks projected on them
    for xp in x_peaks:
        cv2.line(img_color, (xp, 0),(xp, rows - 1), (255, 0, 0), 4, cv2.LINE_AA)
    for yp in y_peaks:
        cv2.line(img_color, (0, yp), (cols-1, yp), (0, 255, 0), 4, cv2.LINE_AA)

    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', img_color)
    cv2.waitKey(0)


