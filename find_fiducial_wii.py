import os
import numpy as np
import cv2
import pickle
from hist_utils import getHSVHistogram, histogramHSVPlot


class fiducialFinder:

    def __init__(self, cachePath='.'):
        self.channels = [0]
        self.hist_size = [180]
        self.hist_range = [0, 180]
        self.color_histogram_feature_file = cachePath + "/pickels/fiducial.pickle"
        self.histogram_source_file = cachePath + '/images/fiducial.png'
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

        b, g, r = model[:, :, 0], model[:, :, 1], model[:, :, 2]
        bisg = np.equal(b, g)
        gisr = np.equal(g, r)
        is_gray = np.logical_and(bisg, gisr)
        is_not_gray = np.invert(is_gray)
        white = np.full(b.shape, 255)
        is_white = np.equal(b, white)
        mask = np.logical_and(is_gray, is_white)
        #mask = np.logical_and(mask, is_not_gray)
        cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
        cv2.imshow('Display', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        """Generate the histogram to using the hist type indicated in the initialization

        @param model_frame the frame to add to the model, its histogram
            is obtained and saved in internal list.
        """
        ht = getHSVHistogram(model, mask*255)
        histogramHSVPlot(ht)

        hsv_image = cv2.cvtColor(model, cv2.COLOR_BGR2HSV)
        channels = cv2.split(hsv_image)



        hist = cv2.calcHist([hsv_image], self.channels, channels[1], self.hist_size, self.hist_range)
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

    def check(self, image_in):

        hsv_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2HSV)
        hsv_image = self.preprocessHSV(hsv_image)

        image_hist = cv2.calcHist([hsv_image], self.channels, None, self.hist_size, self.hist_range)
      #  cv2.imshow('image_hist', image_hist)
      #  cv2.waitKey(0);

        image_hist = cv2.normalize(image_hist, image_hist).flatten()
        y = cv2.compareHist(self.hist, image_hist, cv2.HISTCMP_INTERSECT)
        y = np.log(y)
        y = 1.0 / np.fabs(y)
        self.intersection.append(y)
        print("[INFO] (%d,%f)" % (len(self.intersection), y))

        # use normalized histogram and apply backprojection
        dst = cv2.calcBackProject([hsv_image], self.channels, self.bphist, self.hist_range, 1)

        # Now convolute with circular disc
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cv2.filter2D(dst, -1, disc, dst)

        # threshold and binary AND
        median = np.median(dst)
        ret, thresh = cv2.threshold(dst, median, 255, 0)
        thresh_bgr = cv2.merge((thresh, thresh, thresh))
        res = cv2.bitwise_and(image_in, thresh_bgr)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        return y,thresh, res, cnts

    def polyLinePoints(self, width, height):
        vals = self.history()
        maxi = np.max(vals)
        x = np.arange(len(vals))
        vals = np.multiply(height, vals)
        vals = np.subtract(height, vals)
        pts = np.vstack((x, vals)).astype(np.int32).T
        return pts


