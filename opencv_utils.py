# !/usr/bin/python3
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# pp = Path(os.getcwd() + '/../pairopy')
# sys.path.append(str(pp))


def mouse_handler(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(data['im'], (x, y), 3, (0, 0, 255), 5, 16);
        cv2.imshow("Image", data['im']);
        if len(data['points']) < 4:
            data['points'].append([x, y])


def get_four_points(im):
    # Set up data to send to mouse handler
    data = {}
    data['im'] = im.copy()
    data['points'] = []

    # Set the callback function for any mouse event
    cv2.imshow("Image", im)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)

    # Convert array to np.array
    points = np.vstack(data['points']).astype(float)

    return points


def skiimage_display(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fetch_reduce_image_from_file(filename, reduce):
    ## Note image read by skimage and therefore RGB
    bgr = cv2.imread(filename)
    h, w, channels = bgr.shape

    if reduce == 1:
        return bgr
    frame = (int(w / reduce), int(h / reduce))
    bgr_clone = cv2.resize(bgr, frame)
    return bgr_clone


def load_reduce_convert(image_file, reduce):
    bgr_image = fetch_reduce_image_from_file(image_file, reduce)
    h, w, channels = bgr_image.shape
    if channels == 3:
        lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        # Split LAB channels
        L, a, b = cv2.split(lab_image)
        return (L, a, b)
    if channels == 4:
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGRA2BGR)
        lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        # Split LAB channels
        L, a, b = cv2.split(lab_image)
        return (L, a, b)
    if channels == 1:
        return bgr_image


'''
Parameters: image file path and reduction factor
Returns a dictionary reduced and full_shape
'''


def import_reduce_from_file(filename, reduce):
    ## Note image read by skimage and therefore RGB
    bgr = cv2.imread(filename)
    h, w, channels = bgr.shape

    if reduce == 1:
        return bgr
    frame = (int(w / reduce), int(h / reduce))
    bgr_clone = cv2.resize(bgr, frame)
    return {'reduced': bgr_clone, 'full_shape': bgr.shape}


'''
Parameters: image file and reduction factor
Returns a dictionary of full_shape, reduced and LAB for reduced
@todo: Add choice to which one to convert and cache
'''


def import_reduce_convertLAB(image_file, reduce):
    im = import_reduce_from_file(image_file, reduce)
    bgr_image = im['reduced']
    h, w, channels = bgr_image.shape
    if channels == 3:
        lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        # Split LAB channels
        L, a, b = cv2.split(lab_image)
        im['LAB'] = (L, a, b)
    if channels == 4:
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGRA2BGR)
        lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        # Split LAB channels
        L, a, b = cv2.split(lab_image)
        im['LAB'] = (L, a, b)
    if channels == 1:
        im['LAB'] = None
    return im


def import_reduce_convert(image_file, reduce):
    im = import_reduce_from_file(image_file, reduce)
    bgr_image = im['reduced']
    h, w, channels = bgr_image.shape
    if channels == 4:
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGRA2BGR)
        im['reduced'] = bgr_image

    return im


def image_stats(image):
    # compute the mean and standard deviation of each channel
    channels = cv2.split(image)
    stats = []
    for c in channels:
        mean_std_list = (c.mean(), c.std())
        stats.append(mean_std_list)
    return stats


def convert_lab2bgr(lab_tuple):
    merged = cv2.merge(lab_tuple)
    return cv2.cvtColor(merged.astype("uint8"), cv2.COLOR_LAB2BGR)


def convert_lab2rgb(lab_tuple):
    merged = cv2.merge(lab_tuple)
    return cv2.cvtColor(merged.astype("uint8"), cv2.COLOR_LAB2RGB)


def lab_image_stats(image):
    """
    Parameters:
    -------
    image: NumPy array
        OpenCV image in L*a*b* color space
    Returns:
    -------
    Tuple of mean and standard deviations for the L*, a*, and b*
    channels, respectively
    """
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)


def color_transfer(source, target):
    """
    Transfers the color distribution from the source to the target
    image using the mean and standard deviations of the L*a*b*
    color space.
    This implementation is (loosely) based on to the "Color Transfer
    between Images" paper by Reinhard et al., 2001.
    Parameters:
    -------
    source: NumPy array
        OpenCV image in BGR color space (the source image)
    target: NumPy array
        OpenCV image in BGR color space (the target image)
    Returns:
    -------
    transfer: NumPy array
        OpenCV image (w, h, 3) NumPy array (uint8)
    """
    # convert the images from the RGB to L*ab* color space, being
    # sure to utilizing the floating point data type (note: OpenCV
    # expects floats to be 32-bit, so use that instead of 64-bit)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # compute color statistics for the source and target images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = lab_image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = lab_image_stats(target)

    # subtract the means from the target image
    (l, a, b) = cv2.split(target)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    # scale by the standard deviations
    l = (lStdTar / lStdSrc) * l
    a = (aStdTar / aStdSrc) * a
    b = (bStdTar / bStdSrc) * b

    # add in the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    # clip the pixel intensities to [0, 255] if they fall outside
    # this range
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    # merge the channels together and convert back to the RGB color
    # space, being sure to utilize the 8-bit unsigned integer data
    # type
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)  # return the color transferred image
    return transfer


def lab_image_stats(image):
    """
    Parameters:
    -------
    image: NumPy array
        OpenCV image in L*a*b* color space
    Returns:
    -------
    Tuple of mean and standard deviations for the L*, a*, and b*
    channels, respectively
    """
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)


def invertLABluminance(bgr_image):
    in_lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    (l, a, b) = cv2.split(in_lab)
    l = 255 - l
    outlab = cv2.merge([l, a, b])
    outbgr = cv2.cvtColor(outlab, cv2.COLOR_LAB2BGR)
    return outbgr


if __name__ == '__main__':
    name = sys.argv[1]
    img_color = cv2.imread(name)

    outbgr = invertLABluminance(img_color)

    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', outbgr)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
