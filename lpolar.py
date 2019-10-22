#!/usr/bin/env python

'''
plots image as logPolar and linearPolar

Usage:
    logpolar.py

Keys:
    ESC    - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv

if __name__ == '__main__':
    print(__doc__)

    import sys

    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 'fruits.jpg'

    img = cv.imread(fn)
    if img is None:
        print('Failed to load image file:', fn)
        sys.exit(1)

    img3 = cv.linearPolar(img, (314, 169), 20, cv.WARP_FILL_OUTLIERS + cv.INTER_LINEAR)

    cv.imshow('before', img)
    cv.imshow('linearpolar', img3)
    cv.imwrite('/Users/arman/Desktop/linear.png', img3)

    cv.waitKey(0)
