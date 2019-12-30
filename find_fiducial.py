#!/usr/bin/env python
import cv2
import numpy as np
import matplotlib as plt
import math
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
from numpy.random import rand, randint, randn
from color_segmentation import color_cluster_seg

class FindFiducial():
    def __init__(self, image):
        self.image = image
        shape = image.shape
        self.frame_width = shape[1]
        self.frame_height = shape[0]


        # Image variables for displaying data
        self.img = None
        self.canny = None

        self.find_squares()

        if not (self.img is None):
            cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
            cv2.imshow('Display', self.img)
            cv2.namedWindow('canny', cv2.WINDOW_NORMAL)
            cv2.imshow('canny', self.canny)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def find_squares(self):
        rng = np.random.RandomState(1234)
        img = self.image
        img_display = img
        #img = cv2.GaussianBlur(img, (5, 5), 0)
        #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, (9, 9))

        squares = []
        img = cv2.Canny(img, 1, 100, apertureSize=3)
        self.canny = img
        img = cv2.dilate(img, None)
        retval, img = cv2.threshold(img, 1, 255, cv2.THRESH_OTSU and cv2.THRESH_BINARY)
        image, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours_poly = []
        boundRect = []
        cc = 0
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            is_convex = cv2.isContourConvex(c)
            if area < 100: #or (not is_convex):
                continue

            poly = cv2.approxPolyDP(c, 3, True)
            bb = cv2.boundingRect(poly)
            mind = min(bb[2],bb[3])
            maxd = max(bb[2],bb[3])
            ar = maxd // mind
            if ar < 10: continue
            cc += 1
            contours_poly.append(poly)
            boundRect.append(cv2.boundingRect(poly))

            # Draw polygonal contour + bonding rects + circles
        for i in range(cc):
            color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
            cv2.drawContours(img_display, contours_poly, i, color)
            cv2.rectangle(img_display, (int(boundRect[i][0]), int(boundRect[i][1])), \
                         (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
         #   cv2.circle(img_display, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)

        self.img = img_display

    def is_square(self, cnt, epsilon):
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, epsilon * cnt_len, True)

        # if len(cnt) != 4 or not cv2.isContourConvex(cnt):
        #     return (cnt, False)
        # else:
        cnt = cnt.reshape(-1, 2)
        count = len(cnt)
        max_cos = np.max([self.angle_cos(cnt[i], cnt[(i + 1) % count], cnt[(i + 2) % count]) for i in range(count)])

        return (cnt, max_cos < 0.1)

    def angle_cos(self, p0, p1, p2):
        d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
        return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


if __name__ == '__main__':
    if not Path(sys.argv[1]).is_file() or not Path(sys.argv[1]).exists():
        print(sys.argv[1] + '  Does not exist ')
    lab_tuple = opencv_utils.load_reduce_convert(sys.argv[1], 2)

    var = FindFiducial(lab_tuple[0])
