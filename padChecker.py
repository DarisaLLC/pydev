import os
import numpy as np
import cv2
import sys
from pathlib import Path
import pickle

from matplotlib import pyplot as plt
from color_histogram_classifier import HistogramColorClassifier


class padChecker:

    def __init__(self, cachePath='./projects/wiic'):

        self.intersection = list()

        # Defining the classifier
        self.classifier = HistogramColorClassifier(channels=[0, 1, 2], hist_size=[128, 128, 128],
                                                 hist_range=[0, 256, 0, 256, 0, 256], hist_type='BGR')

        self.color_histogram_feature_file = cachePath + "/pickels/pad.pickle"
        self.histogram_source_file = cachePath + '/images/pad.png'
        self.hist = None
        if (not os.path.exists(self.color_histogram_feature_file)):
            self.model = cv2.imread(self.histogram_source_file)  # Pad
            self.hist = self.classifier.generateModelHistogram(self.model)
            print("[INFO] serializing ...")
            f = open(self.color_histogram_feature_file, "wb")
            f.write(pickle.dumps(self.hist))
            f.close()
        else:
            print("[INFO] loading model histogram...")
            self.hist = pickle.loads(open(self.color_histogram_feature_file, "rb").read())

        if self.hist is None:
            print("[ERROR] failed to create or rule loading model histogram...")
        else:
            self.classifier.addModelByHistogram(self.hist)
            print("[INFO] Model Successfully added ...")
        self.clear()

    def clear(self):
        self.intersection = []
        print("[INFO] History Cleared ...")

    def history(self):
        return self.intersection

    def check(self, frame):
        comparison_array = self.classifier.returnHistogramComparisonArray(frame, method="intersection")
        y = np.log(comparison_array[0])
        y = 1.0 / np.fabs(y)
        self.intersection.append(y)
        print("[INFO] (%d,%f)" % (len(self.intersection), y))
        return y






