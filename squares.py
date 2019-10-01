#!/usr/bin/env python

'''
Simple "Square Detector" program.
Loads several images sequentially and tries to find squares in each image.
'''

# Python 2/3 compatibility
import sys
from matplotlib import pyplot as plt

PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

# import sys
# sys.stdout = open('log.txt', 'w')

import functools
import numpy as np
import cv2 as cv

def compare(sq1, sq2):
    y1, x1, h1, w1 = sq1
    y2, x2, h2, w2 = sq2

    if(abs(y1 - y2) < 100 and abs(x1 - x2) < 100):
        return (h1 * w1 - h2 * w2)
    if(abs(y1 - y2) >= 100):
        return 100 * (y1 - y2)
    if(abs(x1 - x2) >= 100):
        return 100 * (x1 - x2)
    return (h1 * w1 - h2 * w2)


def sort_squares(sq_list, img):
    if(len(sq_list) == 0):
        return
    sorted_sq_list = sorted(sq_list, key=functools.cmp_to_key(compare))
    serial = 0
    prev_sq = None
    for sq in sorted_sq_list:
        y, x, h, w = sq

        if prev_sq is not None:
            py, px, ph, pw = prev_sq
            if(abs(y - py) < 100 and abs(x - px) < 100):
                continue
            else:
                prev_sq = sq
        else:
            prev_sq = sq

        if serial < 10:
            serial += 1
            continue

        digit = img[y:y+h, x:x+w]
        print(str(serial - 10) + '.png', y, x, h, w)
        cv.imwrite('./data/' + str(serial - 10) + '.png', digit)
        serial += 1


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
    sq_list = []
    img = cv.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv.Canny(gray, 0, 50, apertureSize=5)
                bin = cv.dilate(bin, None)
            else:
                _retval, bin = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)
            bin, contours, _hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv.arcLength(cnt, True)
                cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
                area = cv.contourArea(cnt)
                is_convex = cv.isContourConvex(cnt)
                if len(cnt) == 4 and  area > 30 and area < 50000 and is_convex:
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        x, y, w, h = cv.boundingRect(cnt)
                        sq_list.append((y, x, h, w))
                        squares.append(cnt)
    sort_squares(sq_list, img)
    return squares

if __name__ == '__main__':
    from glob import glob

    fig = plt.figure(figsize=(22, 15))

    for fn in glob('/Volumes/medvedev/_SP/2019_09_06_BMWi3_natika_camera_images/empty_pad/empty_pad3/*.png'):
        img = cv.imread(fn)
        squares = find_squares(img)


        ax2 = plt.subplot2grid((3, 3), (0, 0))
        ax2.imshow(img, cmap='gray')
        ax2.set_title('Image')
        ax2.plot(squares[:, 0], squares[:, 1], '+g', markersize=3)
        ax2.set_title('Image + CORNERS ')

        plt.show()