#!/usr/bin/env python

'''
Simple "Square Detector" program.

Loads several images sequentially and tries to find squares in each image.
'''

# Python 2/3 compatibility
import sys
import rectangle

PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2 as cv
from pathlib import PurePath


def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


def find_squares(img):
    img = cv.GaussianBlur(img, (3, 3), 0)
    squares = []
    ih, iw = img.shape[:2]
    inside = [0, 0, iw, ih]
    iou_limit = 0.75
    for gray in cv.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv.Canny(gray, 0, 15, apertureSize=5, L2gradient=True)
            else:
                bin = cv.dilate(bin, None)
        _retval, bin = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)
        bin, contours, _hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cnt_len = cv.arcLength(cnt, True)
            cnt = cv.approxPolyDP(cnt, 0.02 * cnt_len, True)
            x, y, w, h = cv.boundingRect(cnt)
            bbox = [x, y, x + w, y + h]
         #   iou = rectangle.intersection_over_union(inside, bbox)
         #   if iou > iou_limit: continue
            if len(cnt) > 4 and cv.contourArea(cnt) > 100 and cv.isContourConvex(cnt):
                cnt = cnt.reshape(-1, 2)
                max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in xrange(4)])
                if max_cos < 0.1:
                    squares.append(cnt)


    return squares

if __name__ == '__main__':
    from glob import glob

    for fn in glob('/Volumes/medvedev/_SP/2019_09_06_BMWi3_natika_camera_images/empty_pad/empty_pad3/*.png'):
        img = cv.imread(fn)
        name = PurePath(fn).parts[1]
        rows, cols, channels = map(int, img.shape)

        #img = cv.pyrDown(img, dstsize=(cols // 2, rows // 2))
        #cols = cols // 2
        #rows = rows // 2
        #      img = cv.pyrDown(img, dstsize=(cols // 2, rows // 2))
        print('img dims: - %d  %d ' % (cols, rows))

        squares = find_squares(img)
        cv.drawContours(img, squares, -1, (0, 255, 0), 1)
        cv.imshow(name, img)
        ch = cv.waitKey()
        if ch == 27:
            break
    cv.destroyAllWindows()
