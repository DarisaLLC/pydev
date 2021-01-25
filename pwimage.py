#!/usr/bin/python3
import os
import sys
from pathlib import Path

import cv2
import numpy as np

pp = Path(os.getcwd() + '/../pairopy')
sys.path.append(str(pp))


class icompare:
    def func(self, image_a, image_b, options): pass


class incv(icompare):
    def func(self, image_a, image_b, options):
        res = cv2.matchTemplate(image_a, image_b, cv2.TM_CCOEFF_NORMED)
        # @todo just fetch[0,0]
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        return max_val


class ichcmp(icompare):
    def func(self, image_a, image_b, options):
        res = cv2.matchTemplate(image_a, image_b, cv2.TM_CCOEFF_NORMED)
        # @todo just fetch[0,0]
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        return max_val


class pwimage:

    def __init__(self, image_list, tag_list):
        self._image_list = image_list
        self._tag_list = tag_list

    def newPairWiseArray(dims, info='information'):
        """Make a matrix with all zeros and increasing elements on the diagonal
        :rtype: numpy.ndarray
        """
        _unity = 1.0

        data = np.zeros(dims)
        for i in range(min(dims)):
            data[i, i] = _unity

        return pwiser.UnityPairWiseArray(data)

    def setPairWiseArrayPair(data, row, col, p):
        """Set both P(i,j) to p and P(j,i) to 1 - p """
        assert (data.info == 'leo-pairwise')
        assert (p >= 0.0 and p <= 1.0)
        data[row, col] = p
        data[col, row] = 1.0 - p
