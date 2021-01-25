import os
import numpy as np
import cv2
import sys
from pathlib import Path
import pickle
from imutils import contours
from matplotlib import pyplot as plt
from skimage.util import img_as_ubyte


def invertLABluminance(bgr_image):
    in_lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    (l, a, b) = cv2.split(in_lab)
    l = 255 - l
    outlab = cv2.merge([l, a, b])
    outbgr = cv2.cvtColor(outlab, cv2.COLOR_LAB2BGR)
    return outbgr


class padChecker:

    def __init__(self, cachePath='.'):
        self.channels = [0, 1]
        self.hist_size = [180, 256]
        self.hist_range = [0, 179, 0, 255]
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


    def generateModelHistogram(self, model):
        """Generate the histogram to using the hist type indicated in the initialization
        @param model_frame the frame to use for the histogram.
            is obtained and saved in internal list.
        """
        hsv_image = cv2.cvtColor(model, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_image], self.channels, None, self.hist_size, self.hist_range)
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

    def check(self, image_in, mask, min_area = 1000, h_score_thr = 0.8, vertical_thr = 0.5):

        hsv_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2HSV)
        image_hist = cv2.calcHist([hsv_image], self.channels, mask, self.hist_size, self.hist_range)
        image_hist = cv2.normalize(image_hist, image_hist).flatten()
        h_score = cv2.compareHist(self.hist, image_hist, cv2.HISTCMP_INTERSECT)
        self.intersection.append(h_score)
        height, width, channels = image_in.shape
        
        # use normalized histogram and apply backprojection
        dst = cv2.calcBackProject([hsv_image], self.channels, self.bphist, self.hist_range, 1)
        dst = cv2.bitwise_and(dst, mask)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
        cv2.filter2D(dst, -1, disc, dst)

        # threshold and binary AND
        ret, thresh = cv2.threshold(dst, 1, 255,  cv2.THRESH_OTSU )
        thresh = cv2.bitwise_and(thresh, mask)
        bounds = []
        rects = []
        scores = []
        directions = []
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        height_bar = height * vertical_thr
        for contour in cnts:
            rr = cv2.minAreaRect(contour)
            bb = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            roi = cv2.boundingRect(contour)
            if roi[0] < 0 or roi[1] < 0:
                continue
            if area < min_area:
                continue
            # if the center is above our valid area skip it
            if rr[0][1] > height_bar:
                continue

            (x, y, w, h) = roi
            roi_hist = cv2.calcHist([hsv_image[y:y+h,x:x+w]], self.channels, mask[y:y+h,x:x+w],
                                      self.hist_size, self.hist_range)
            roi_hist = cv2.normalize(roi_hist, roi_hist).flatten()
            h_score = cv2.compareHist(self.hist, roi_hist, cv2.HISTCMP_INTERSECT)
            if h_score < h_score_thr: continue
            scores.append(h_score)
            bounds.append(bb)
            rects.append(rr)

        order = np.argsort(scores)
        thresh_bgr = cv2.merge((thresh, thresh, thresh))
        res = cv2.bitwise_and(image_in, thresh_bgr)
        return h_score,thresh, res, scores, bounds, rects, order

def region_of_interest(img):
    rows, cols, channels = map(int, img.shape)
    mask = np.zeros((rows, cols), dtype=np.uint8)
    left = int(cols / 2.5)
    right = int(cols - left)

    triangle = np.array([[
        (left, 0),(right, 0), (cols-1, rows - 1),(0, rows-1) ]], np.int32)

    cv2.fillConvexPoly(mask, triangle, 255)
    return mask


if __name__ == '__main__':

    # Equalize Histogram of Color Images
    def equalize_histogram_color(img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img


    def invertLABluminance(bgr_image):
        in_lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        (l, a, b) = cv2.split(in_lab)
        l = 255 - l
        outlab = cv2.merge([l, a, b])
        outbgr = cv2.cvtColor(outlab, cv2.COLOR_LAB2BGR)
        return outbgr


    def detect_ga(frame, mask, checker):
        display = frame.copy()
        y, mask, seethrough, cnts, hulls = checker.check(frame, mask)
        res = np.vstack((display, seethrough))
        return (display, res, y, mask, seethrough, cnts, hulls)


    def crop_image_mask_equalize(image):
        rows, cols, channels = map(int, image.shape)
        # tl = (6, 53)
        # br = (709, 362)
        # frame = image[tl[1]: br[1], tl[0]: br[0]]
        frame = image
        mask = region_of_interest(frame)
        runi = frame.copy()
        runih = equalize_histogram_color(runi)
        return runih, mask

    def process (fqfn, checker):
        img = cv2.imread(fqfn)
        checker = padChecker(cache_path)
        frame, mask = crop_image_mask_equalize(img)
        all = detect_ga(frame, mask, checker)
        print((fqfn, all[2]))
        mmm = all[1] #cv2.merge([all[2], all[2], all[2]])
        im = cv2.drawContours(mmm, all[6], -1, (0, 255, 0), 3)
        return im,all[2]

    if len(sys.argv) < 2:
        sys.exit(1)

    def get_frame_number(fqfn):
        return int(str.split(Path(fqfn).name,'.')[0])

    if Path(sys.argv[1]).is_file():
        file_folder = os.path.dirname(os.path.realpath(__file__)) + '/projects/wiic/'
        cache_path = file_folder
        checker = padChecker(cache_path)
        img = cv2.imread(sys.argv[1])
        frame_number = 0 #get_frame_number(sys.argv[1])
        print(('frame ', frame_number))
        frame, mask = crop_image_mask_equalize(img)
        all = detect_ga(frame, mask, checker)
        mmm = all[1]  # cv2.merge([all[0], all[0], all[0]])
        im = cv2.drawContours(mmm, all[6], -1, (0, 255, 0), 3)
        print((sys.argv[1], all[2]))
        cv2.namedWindow('PadChecker', cv2.WINDOW_NORMAL)
        cv2.imshow('PadChecker', all[4])
        cv2.namedWindow('PadChecker(1)', cv2.WINDOW_NORMAL)
        cv2.imshow('PadChecker(1)', im)
        cv2.waitKey(0)
        all = detect_ga(all[4], mask, checker)
        mmm = all[1]  # cv2.merge([all[0], all[0], all[0]])
        im = cv2.drawContours(mmm, all[6], -1, (0, 255, 0), 3)
        print((sys.argv[1], all[2]))


    elif Path(sys.argv[1]).is_dir():
        output_dir = '/Users/arman/tmpout/'
        file_folder = os.path.dirname(os.path.realpath(__file__)) + '/projects/wiic/'
        checker = padChecker(file_folder)
        cache_path = file_folder
        scores = {}
        for file in os.listdir(sys.argv[1]):
            if file.endswith(".png"):
                fqfn = os.path.join(sys.argv[1], file)
                fn = get_frame_number(fqfn)
                out_img, score = process(fqfn, checker)
                scores[fn] = score
                ofqfn = os.path.join(output_dir,file)
                cv2.imwrite(ofqfn, out_img)
                print((fqfn,ofqfn,' Done'))

        print (scores)



