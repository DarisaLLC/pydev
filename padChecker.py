import os
import numpy as np
import cv2
import sys
from pathlib import Path
import pickle
from imutils import contours
from matplotlib import pyplot as plt

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

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        return y,thresh, res, cnts

    def polyLinePoints(self, width, height):
        vals = self.history()
        maxi = np.max(vals)
        x = np.arange(len(vals))
        vals = np.multiply(height, vals)
        vals = np.subtract(height, vals)
        pts = np.vstack((x, vals)).astype(np.int32).T
        return pts

class ColorClassifier:
    """Classifier for comparing an image I with a model M. The comparison is based on color
    histograms. It included an implementation of the Histogram Intersection algorithm.


    """

    def __init__(self, channels=[0, 1], hist_size=[180, 256], hist_range=[0, 180, 0, 256], hist_type='HSV'):
        """Init the classifier.

        This class has an internal list containing all the models.
        it is possible to append new models. Using the default values
        it extracts a 3D BGR color histogram from the image, using
	10 bins per channel.
        @param channels list where we specify the index of the channel
           we want to compute a histogram for. For a grayscale image,
           the list would be [0]. For all three (red, green, blue) channels,
           the channels list would be [0, 1, 2].
        @param hist_size number of bins we want to use when computing a histogram.
            It is a list (one value for each channel). Note: the bin sizes can
            be different for each channel.
        @param hist_range it is the min-max value of the values stored in the histogram.
            For three channels can be [0, 256, 0, 256, 0, 256], if there is only one
            channel can be [0, 256]
        @param hsv_type Convert the input BGR frame in HSV or GRAYSCALE. before taking
            the histogram. The HSV representation can get more reliable results in
            situations where light have a strong influence.
            BGR: (default) do not convert the input frame
            HSV: convert in HSV represantation
            GRAY: convert in grayscale
        """
        self.channels = channels
        self.hist_size = hist_size
        self.hist_range = hist_range
        self.hist_type = hist_type
        self.model_list = list()
        self.name_list = list()



    def addModelByHistogram(self, hist, name=''):
        """Add the histogram to internal container. If the name of the object
           is already present then replace that histogram with a new one.

        @param hist to add to the model, its histogram
            is obtained and saved in internal list.
        @param name a string representing the name of the model.
            If nothing is specified then the name will be the index of the element.
        """
        if name == '': name = str(len(self.model_list))
        if name not in self.name_list:
            self.model_list.append(hist)
            self.name_list.append(name)
        else:
            for i in range(len(self.name_list)):
                if self.name_list[i] == name:
                    self.model_list[i] = hist
                    break

    def addModelByFrame(self, model_frame, name=''):
        """Add the histogram to internal container. If the name of the object
           is already present then replace that histogram with a new one.

        @param model_frame the frame to add to the model, its histogram
            is obtained and saved in internal list.
        @param name a string representing the name of the model.
            If nothing is specified then the name will be the index of the element.
        """
        hist = self.generateModelHistogram(model_frame)
        if name == '': name = str(len(self.model_list))
        if name not in self.name_list:
            self.model_list.append(hist)
            self.name_list.append(name)
        else:
            for i in range(len(self.name_list)):
                if self.name_list[i] == name:
                    self.model_list[i] = hist
                    break

    def removeModelHistogramByName(self, name):
        """Remove the specific model using the name as index.

        @param: name the index of the element to remove
        @return: True if the object has been deleted, otherwise False.
        """
        if name not in self.name_list:
            return False
        for i in range(len(self.name_list)):
            if self.name_list[i] == name:
                del self.name_list[i]
                del self.model_list[i]
                return True

    def returnHistogramComparison(self, hist_1, hist_2, method='intersection'):
        """Return the comparison value of two histograms.

        Comparing an histogram with itself return 1.
        @param hist_1
        @param hist_2
        @param method the comparison method.
            intersection: (default) the histogram intersection (Swain, Ballard)
        """
        assert(cv2.__version__.split(".")[0] == '3')
        if(method=="intersection"):
            comparison = cv2.compareHist(hist_1, hist_2, cv2.HISTCMP_INTERSECT)
        elif(method=="correlation"):
            comparison = cv2.compareHist(hist_1, hist_2, cv2.HISTCMP_CORREL)
        elif(method=="chisqr"):
            comparison = cv2.compareHist(hist_1, hist_2, cv2.HISTCMP_CHISQR)
        elif(method=="bhattacharyya"):
            comparison = cv2.compareHist(hist_1, hist_2, cv2.HISTCMP_BHATTACHARYYA)
        else:
            raise ValueError('[DarisaLLC] color_classification.py: the method specified ' + str(method) + ' is not supported.')


        return comparison

    def returnHistogramComparisonArray(self, image_in, method='intersection'):
        """Return the comparison array between all the model and the input image.

        The highest value represents the best match.
        @param image the image to compare
        @param method the comparison method.
            intersection: (default) the histogram intersection (Swain, Ballard)
        @return a numpy array containg the comparison value between each pair image-model
        """
        if(self.hist_type=='HSV'): image = cv2.cvtColor(image_in, cv2.COLOR_BGR2HSV)
        elif(self.hist_type=='GRAY'): image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
        elif(self.hist_type=='RGB'): image = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB)
        comparison_array = np.zeros(len(self.model_list))
        masks = {}
        image_hist = cv2.calcHist([image], self.channels, None, self.hist_size, self.hist_range)
        image_hist = cv2.normalize(image_hist, image_hist).flatten()
        counter = 0
        for model_hist in self.model_list:
            comparison_array[counter] = self.returnHistogramComparison(image_hist, model_hist, method=method)
            cv2.normalize(model_hist,model_hist, 0,255,cv2.NORM_MINMAX)
            dst = cv2.calcBackProject([image], self.channels, model_hist, self.hist_range, 1)

            masks[counter] = dst
            counter += 1
        return (comparison_array, masks)

    def returnHistogramComparisonProbability(self, image, method='intersection'):
        """Return the probability distribution of the comparison between
        all the model and the input image. The sum of the elements in the output
        array sum up to 1.

        The highest value represents the best match.
        @param image the image to compare
        @param method the comparison method.
            intersection: (default) the histogram intersection (Swain, Ballard)
        @return a numpy array containg the comparison value between each pair image-model
        """
        comparison_array = self.returnHistogramComparisonArray(image=image, method=method)
        #comparison_array[comparison_array < 0] = 0 #Remove negative values
        comparison_distribution = np.divide(comparison_array, np.sum(comparison_array))
        return comparison_distribution

    def returnBestMatchIndex(self, image, method='intersection'):
        """Return the index of the best match between the image and the internal models.

        @param image the image to compare
        @param method the comparison method.
            intersection: (default) the histogram intersection (Swain, Ballard)
        @return a numpy array containg the comparison value between each pair image-model
        """
        comparison_array = self.returnHistogramComparisonArray(image, method=method)
        return np.argmax(comparison_array)

    def returnBestMatchName(self, image, method='intersection'):
        """Return the name of the best match between the image and the internal models.

        @param image the image to compare
        @param method the comparison method.
            intersection: (default) the histogram intersection (Swain, Ballard)
        @return a string representing the name of the best matching model
        """
        comparison_array = self.returnHistogramComparisonArray(image, method=method)
        arg_max = np.argmax(comparison_array)
        return self.name_list[arg_max]

    def returnNameList(self):
        """Return a list containing all the names stored in the model.

        @return: a list containing the name of the models.
        """
        return self.name_list

    def returnSize(self):
        """Return the number of elements stored.

        @return: an integer representing the number of elements stored
        """
        return len(self.model_list)




