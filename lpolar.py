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

import cv2

if __name__ == '__main__':
    print(__doc__)

    import sys

    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 'fruits.jpg'

    img = cv2.imread(fn)
    if img is None:
        print('Failed to load image file:', fn)
        sys.exit(1)

    window = img[1000:1400,2000:2400]
    img3 = cv2.linearPolar(window, (200,200), 200, cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR)

    cv2.namedWindow('before', cv2.WINDOW_NORMAL)
    cv2.imshow('before', window)
    cv2.namedWindow('linearpolar', cv2.WINDOW_NORMAL)
    cv2.imshow('linearpolar', img3)
    cv2.imwrite('/Users/arman/Desktop/linear.png', img3)

    cv2.waitKey(0)
