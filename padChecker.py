import os
import numpy as np
import cv2
import sys
from pathlib import Path
import pickle
from imutils import contours
from matplotlib import pyplot as plt
from skimage.util import img_as_ubyte

class padChecker:

    def __init__(self, cachePath='.'):
        self.channels = [0]
        self.hist_size = [180]
        self.hist_range = [0, 180]
        self.color_histogram_feature_file = cachePath + "/pickels/pad.pickle"
        self.histogram_source_file = cachePath + '/images/pad.png'
        self.bphist = None
        self.hist = None
        if (not os.path.exists(self.color_histogram_feature_file)):
            self.model = cv2.imread(self.histogram_source_file)  # Pad
            hists = self.generateModelHistogram(self.model)
            print("[INFO] serializing ...")
            f = open(self.color_histogram_feature_file, "wb")
            f.write(pickle.dumps(hists))
            f.close()
            self.hist = hists[0]
            self.bphist = hists[1]
        else:
            print("[INFO] loading model histogram...")
            hists = pickle.loads(open(self.color_histogram_feature_file, "rb").read())
            self.hist = hists[0]
            self.bphist = hists[1]

        if (self.hist is None) or (self.bphist is None):
            print("[ERROR] failed to create or rule loading model histogram...")

        print("[INFO] Model Successfully added ...")
        self.clear()

    def PrintImage(self, image, nRows, nCols):
        for i in range(nRows):
            print("[%d]" % i, end=" ")
            for j in range(nCols):
                print(" %d" % image[i][j], end=" ")
            print('\n')

    def preprocessHSV(self, hsv):
        hue, sat, vol = cv2.split(hsv)
        kernel = 5
        hue = cv2.medianBlur(hue, kernel, hue)
        sat = cv2.medianBlur(sat, kernel, sat)
        return cv2.merge((hue, sat, vol))

    def smoothHistogram(self, hist_in):
        shape = hist_in.shape
        hist_out = hist_in.copy()
        for i in range(1,shape[0]-1,1):
            hist_out[i] = int((hist_in[i-1]+2*hist_in[i]+hist_in[i+1]) / 4 + 0.5)
        return hist_out

    def generateModelHistogram(self, model):
        """Generate the histogram to using the hist type indicated in the initialization

        @param model_frame the frame to add to the model, its histogram
            is obtained and saved in internal list.
        """
        hsv_image = cv2.cvtColor(model, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_image], self.channels, None, self.hist_size, self.hist_range)
        self.PrintImage(hist, self.hist_size[0], 1)  # , self.hist_size[1])
        hist = self.smoothHistogram(hist)
        self.PrintImage(hist, self.hist_size[0], 1)  # , self.hist_size[1])
        bphist = hist.copy()
        hist = cv2.normalize(hist, hist)

        hist = hist.flatten()
        cv2.normalize(bphist, bphist, 255, cv2.NORM_MINMAX)
        return (hist, bphist)

    def clear(self):
        self.intersection = []
        print("[INFO] History Cleared ...")

    def history(self):
        return self.intersection

    def check(self, image_in, mask):

        hsv_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2HSV)
        hsv_image = self.preprocessHSV(hsv_image)

        image_hist = cv2.calcHist([hsv_image], self.channels, mask, self.hist_size, self.hist_range)

        image_hist = cv2.normalize(image_hist, image_hist).flatten()
        y = cv2.compareHist(self.hist, image_hist, cv2.HISTCMP_BHATTACHARYYA) #  cv2.HISTCMP_INTERSECT
        y = np.log(y)
        y = 1.0 / np.fabs(y)
        self.intersection.append(y)
        print("[INFO] (%d,%f)" % (len(self.intersection), y))

        # use normalized histogram and apply backprojection
        dst = cv2.calcBackProject([hsv_image], self.channels, self.bphist, self.hist_range, 1)
        dst = cv2.bitwise_and(dst, mask)

        # Now convolute with circular disc
        disc = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        cv2.filter2D(dst, -1, disc, dst)

        # threshold and binary AND
        ret, thresh = cv2.threshold(dst, 1, 255,  cv2.THRESH_OTSU )
        hulls = []
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        for contour in cnts:
            hull = cv2.convexHull(contour)
            hulls.append(hull)

        thresh_bgr = cv2.merge((thresh, thresh, thresh))
        res = cv2.bitwise_and(image_in, thresh_bgr)
        return y,thresh, res, cnts, hulls

def region_of_interest(img):
    rows, cols, channels = map(int, img.shape)
    mask = np.zeros((rows, cols), dtype=np.uint8)
    left = int(cols / 2.5)
    right = int(cols - left)

    triangle = np.array([[
        (left, 0),(right, 0), (cols-1, rows - 1),(0, rows-1) ]], np.int32)

    cv2.fillConvexPoly(mask, triangle, 255)
    return mask

import numpy as np
from itertools import combinations

# https://stackoverflow.com/a/13981450

def intersection(s1, s2):
    """
    Return the intersection point of line segments `s1` and `s2`, or
    None if they do not intersect.
    """
    p, r = s1[0], s1[1] - s1[0]
    q, s = s2[0], s2[1] - s2[0]
    rxs = float(np.cross(r, s))
    if rxs == 0: return None
    t = np.cross(q - p, s) / rxs
    u = np.cross(q - p, r) / rxs
    if 0 < t < 1 and 0 < u < 1:
        return p + t * r
    return None

def convex_quadrilaterals(points):
    """
    Generate the convex quadrilaterals among `points`.
    """
    segments = combinations(points, 2)
    for s1, s2 in combinations(segments, 2):
        if intersection(s1, s2) != None:
            yield s1, s2

if __name__ == '__main__':

    # Equalize Histogram of Color Images
    def equalize_histogram_color(img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img

    def detect_ga(frame, mask, checker):
        display = frame.copy()
        frame = cv2.medianBlur(frame, 7)
        y, mask, seethrough, cnts, hulls = checker.check(frame, mask)
        res = np.vstack((display, seethrough))
        return (display, res, y, mask, seethrough, cnts, hulls)


    def crop_image_mask_equalize(image):
        rows, cols, channels = map(int, image.shape)
        tl = (6, 53)
        br = (709, 362)
        frame = image[tl[1]: br[1], tl[0]: br[0]]
        mask = region_of_interest(frame)
        runi = frame.copy()
        runih = equalize_histogram_color(runi)
        return runih, mask

    def process (fqfn):
        file_folder = os.path.dirname(os.path.realpath(__file__)) + '/projects/wiic/'
        cache_path = file_folder
        img = cv2.imread(fqfn)
        checker = padChecker(cache_path)
        frame, mask = crop_image_mask_equalize(img)
        all = detect_ga(frame, mask, checker)
        mmm = cv2.merge([all[3], all[3], all[3]])
        im = cv2.drawContours(mmm, all[6], -1, (0, 255, 0), 3)
        return im

    if len(sys.argv) < 2:
        exit(1)


    if Path(sys.argv[1]).is_file():
        file_folder = os.path.dirname(os.path.realpath(__file__)) + '/projects/wiic/'
        cache_path = file_folder
        checker = padChecker(cache_path)
        img = cv2.imread(sys.argv[1])
        frame, mask = crop_image_mask_equalize(img)
        all = detect_ga(frame, mask, checker)
        mmm = cv2.merge([all[3], all[3], all[3]])
        im = cv2.drawContours(mmm, all[6], -1, (0, 255, 0), 3)

        cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
        cv2.imshow('Display', all[0])
        cv2.namedWindow('PadChecker', cv2.WINDOW_NORMAL)
        cv2.imshow('PadChecker', all[1])
        cv2.namedWindow('PadChecker2', cv2.WINDOW_NORMAL)
        cv2.imshow('PadChecker2', im)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif Path(sys.argv[1]).is_dir():
        output_dir = '/Users/arman/tmpout/'
        for file in os.listdir(sys.argv[1]):
            if file.endswith(".png"):
                fqfn = os.path.join(sys.argv[1], file)
                out_img = process(fqfn)
                ofqfn = os.path.join(output_dir,file)
                cv2.imwrite(ofqfn, out_img)
                print((fqfn,ofqfn,' Done'))




