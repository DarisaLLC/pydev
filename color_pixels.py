# -*- coding: utf-8 -*-
## @package color_histogram.core.color_pixels
#
#  Simple color pixel class.
#
#  @author      tody
#  @date        2015/08/28

import argparse

import os
import sys
from pathlib import Path

pp = Path(os.getcwd() + '/./')
sys.path.append(str(pp))
print(pp)

import numpy as np

import cv_io


## Implementation of color pixels.
#
#  input image is automatically converted into np.float32 format.
class ColorPixels:
    ## Constructor
    #  @param image          input image.
    #  @param num_pixels     target number of pixels from the image.
    def __init__(self, image, num_pixels=1000):
        self._image = cv_io.to32F(image)
        self._num_pixels = num_pixels
        self._rgb_pixels = None
        self._Lab = None
        self._hsv = None

    ## RGB pixels.
    def rgb(self):
        if self._rgb_pixels is None:
            self._rgb_pixels = self.pixels("rgb")
        return self._rgb_pixels

    ## Lab pixels.
    def Lab(self):
        if self._Lab is None:
            self._Lab = self.pixels("Lab")
        return self._Lab

    ## HSV pixels.
    def hsv(self):
        if self._hsv is None:
            self._hsv = self.pixels("hsv")
        return self._hsv

    ## Pixels of the given color space.
    def pixels(self, color_space="rgb"):
        image = np.array(self._image)
        if color_space == "rgb":
            if _isGray(image):
                image = cv_io.gray2rgb(image)

        if color_space == "Lab":
            image = cv_io.rgb2Lab(self._image)

        if color_space == "hsv":
            image = cv_io.rgb2hsv(self._image)
        return self._image2pixels(image)

    def _image2pixels(self, image):
        if _isGray(image):
            h, w = image.shape
            step = h * w / self._num_pixels
            return image.reshape((h * w))[::step]

        h, w, cs = image.shape
        step = int(h * w / self._num_pixels)
        return image.reshape((-1, cs))[::step]


def _isGray(image):
    return len(image.shape) == 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='color_pixels')
    parser.add_argument('--image', '-i', required=True,
                        help='Color Image')
    parser.add_argument('--out', '-o', required=False,
                        help='valid path for output')

    args = parser.parse_args()

    error = not Path(args.image).exists()
    if error:
        print('image file ' + args.image + ' Does not exist ')
        exit(1)

    if args.out:
        error = not Path(args.out).parent.exists()
        if error:
            print('output file ' + args.out + ' Does not exist ')
            exit(1)

    foo = cv_io.loadRGB(args.image)
    foo = ColorPixels(foo)
