# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import cv2
import numpy as np
from skimage import color
from skimage.morphology import square, erosion
from skimage.util import img_as_float
from skimage.util import img_as_ubyte
from skimage import io
from sklearn.cluster import KMeans
from collections import Counter
import cv2  # for resizing image
from skimage import io
from matplotlib import pyplot as plt
from collections import namedtuple


# @memorize.Memorize
def fetch_image_for_image_file(filename):
    return io.imread(filename)

def get_dominant_color(image, k=4, image_processing_size=None):
    """
    takes an image as input
    returns the dominant color of the image as a list

    dominant color is found by running k means on the
    pixels & returning the centroid of the largest cluster

    processing time is sped up by working with a smaller image;
    this resizing can be done with the image_processing_size param
    which takes a tuple of image dims as input

    >>> get_dominant_color(my_image, k=4, image_processing_size = (25, 25))
    [56.2423442, 34.0834233, 70.1234123]
    """
    # resize image if new dims provided
    if image_processing_size is not None:
        image = cv2.resize(image, image_processing_size,
                           interpolation=cv2.INTER_AREA)

    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster and assign labels to the pixels
    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(image)

    # count labels to find most popular
    label_counts = Counter(labels)

    # subset out most popular centroid
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

    return list(dominant_color)

def iter_channels(color_image):
    """Yield color channels of an image."""
    # Roll array-axis so that we iterate over the color channels of an image.
    for channel in np.rollaxis(color_image, -1):
        yield channel


def iter_pixels(image):
    """ Yield pixel position (row, column) and pixel intensity. """
    height, width = image.shape[:2]
    for i in range(height):
        for j in range(width):
            yield (i, j), image[i, j]

def mutual_information(hgram):
    """ Mutual information for joint histogram
    """
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def pct_total_area(image, percentile=0.80):
    """Return threshold value based on percentage of total area.

    The specified percent of pixels less than the given intensity threshold.
    """
    idx = int((image.size - 1) * percentile)
    sorted_pixels = np.sort(image.flat)
    return sorted_pixels[idx]


def stretch_pre(nimg):
    """
    from 'Applicability Of White-Balancing Algorithms to Restoring Faded Colour Slides: An Empirical Evaluation'
    """
    nimg = nimg.transpose(2, 0, 1)
    nimg[0] = np.maximum(nimg[0] - nimg[0].min(), 0)
    nimg[1] = np.maximum(nimg[1] - nimg[1].min(), 0)
    nimg[2] = np.maximum(nimg[2] - nimg[2].min(), 0)
    return nimg.transpose(1, 2, 0)


def gray_world (img):
    image = img_as_float (img)
    dst = image.copy ()
    Red = image [:, :, 0]
    Green = image [:, :, 1]
    Blue = image [:, :, 2]
    means = [np.average (Red), np.average (Green), np.average (Blue)]
    dst [:, :, 0] = np.minimum (Red * (means [1] / means [0]), 1.0)
    dst [:, :, 2] = np.minimum (Blue * (means [1] / means [2]), 1.0)
    return img_as_ubyte (dst)


def omf(image):
    Red = image[:, :, 0]
    Green = image[:, :, 1]
    Blue = image[:, :, 2]
    RL = np.pow(np.add(Red,1),0.3)
    GL = np.pow(np.add(Green,1),0.6)
    BL = np.pow(np.add(Blue,1),0.1)


    omf_r = channel_omf(Red)
    omf_g = channel_omf(Green)
    omf_b = channel_omf(Blue)
    omf_all = (omf_r,omf_g, omf_b)
    print(omf_all)
    return omf_all


def geom_gray(nimg):
    image = img_as_float(nimg)
    Red = image[:, :, 0]
    Green = image[:, :, 1]
    Blue = image[:, :, 2]
    gg = Red * Green * Blue
    return img_as_ubyte(gg)

def mean_gray(nimg):
    image = img_as_float(nimg)
    Red = image[:, :, 0]
    Green = image[:, :, 1]
    Blue = image[:, :, 2]
    gg = (Red + Green + Blue)/ 3
    return img_as_ubyte(gg)


def grey_world_median(nimg):
    image = img_as_float(nimg)
    dst = image.copy()
    Red = image[:, :, 0]
    Green = image[:, :, 1]
    Blue = image[:, :, 2]
    med = [np.median(Red), np.median(Green), np.median(Blue)]
    dst[:, :, 0] = np.minimum(Red * (med[1] / med[0]), 1.0)
    dst[:, :, 2] = np.minimum(Blue * (med[1] / med[2]), 1.0)
    return img_as_ubyte(dst)


def gray_world_median_bgr(nimg):
    image = img_as_float(nimg)
    dst = image.copy()
    Red = image[:, :, 2]
    Green = image[:, :, 1]
    Blue = image[:, :, 0]
    med = [np.median(Red), np.median(Green), np.median(Blue)]
    dst[:, :, 2] = np.minimum(Red * (med[1] / med[0]), 1.0)
    dst[:, :, 1] = np.minimum(Blue * (med[1] / med[2]), 1.0)
    return img_as_ubyte(dst)


def grey_world_max (nimg):
    image = img_as_float (nimg)
    dst = image.copy ()
    Red = image [:, :, 0]
    Green = image [:, :, 1]
    Blue = image [:, :, 2]
    ax = [np.max(Red), np.max(Green), np.max(Blue)]
    dst[:, :, 0] = np.minimum(Red * (ax[1] / ax[0]), 1.0)
    dst[:, :, 2] = np.minimum(Blue * (ax[1] / ax[2]), 1.0)
    return img_as_ubyte (dst)


def get_lut (low):
    # Clamp either end
    if low < -255: low = -255
    if low > 255: low = 255
    base = low
    if base < 0: base += 256
    
    lut = np.linspace (0, 255, 256, dtype=np.uint8)

    for i in range(256):
        lut[i] = (base + i) % 256
    return lut


# Hue is in -180 to + 180 or 0 - 360. in uint8 0 - 255
# angle is >= 0 and < 255.
def rotate_hue (rgb_image, rotate):
    lut = get_lut (rotate)
    hsl = color.convert_colorspace (rgb_image, 'RGB', 'HSV')
    hsl8 = img_as_ubyte (hsl)
    hsl8 [:, :, 0] = lut [hsl8 [:, :, 0]]
    new_rgb = color.convert_colorspace (hsl8, 'HSV', 'RGB')
    return new_rgb


def get_hue (rgb_image):  # , half_bin, step):
    hsl = color.convert_colorspace (rgb_image, 'RGB', 'HSV')
    return img_as_ubyte (hsl [:, :, 0])


def LabJointHistogram(rgb_image):
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    # Split LAB channels
    L, a, b = cv2.split(lab_image)

    print(str(np.min(a)) + '  ' + str(np.average(a)) + '  ' + str(np.max(a)))
    print(str(np.min(b)) + '  ' + str(np.average(b)) + '  ' + str(np.max(b)))

    xedges, yedges = np.linspace(np.min(a), np.max(a), 128), np.linspace(np.min(b), np.max(b), 128)
    hist, xedges, yedges = np.histogram2d(a.flatten(), b.flatten(), (xedges, yedges))
    xidx = np.clip(np.digitize(a.flatten(), xedges), 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(b.flatten(), yedges), 0, hist.shape[1] - 1)
    c = hist[xidx, yidx]
    return (a.flatten(), b.flatten(), c)


def LabMutualInformation(rgb_image, plotter=None):
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    # Split LAB channels
    L, a, b = cv2.split(lab_image)
    hist_2d, x_edges, y_edges = np.histogram2d(a.ravel(), b.ravel(), bins=20)
    mu = mutual_information(hist_2d)
    if plotter != None:
        hist_2d_log = np.zeros(hist_2d.shape)
        non_zeros = hist_2d != 0
        hist_2d_log[non_zeros] = np.log(hist_2d[non_zeros])
        plotter.imshow(hist_2d_log.T, origin='lower')
    return mu


def MutualInformation(a, b, plotter=None):
    hist_2d, x_edges, y_edges = np.histogram2d ( a.ravel (), b.ravel (), bins=20 )
    mu = mutual_information ( hist_2d )
    if plotter != None:
        hist_2d_log = np.zeros ( hist_2d.shape )
        non_zeros = hist_2d != 0
        hist_2d_log[non_zeros] = np.log ( hist_2d[non_zeros] )
        plotter.imshow ( hist_2d_log.T, origin='lower' )
    return mu


def CenterOfIntensity(gray):
    moments = cv2.moments(gray)  # Calculate moments
    (cx, cy) = (-1.0, -1.0)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
        cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
    return (cx, cy)


def LabDirection(bgr_image, plotter=None):
    lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    # Split LAB channels
    L, a, b = cv2.split(lab_image)
    dir = cv2.fastAtan2(b, a)
    return dir


def DistanceToWhite(ccc_image, islab=False):
    ''' White is 255,255,255 in BGR and 255,0,0 in Lab
     Not handling rgb passed in as tuple
     '''
    if (isinstance(ccc_image, tuple)):
        assert (islab)

    if isinstance(ccc_image, tuple) and islab is True:
        L, a, b = ccc_image
    elif islab is True:
        L, a, b = cv2.split(ccc_image)
    elif not islab:
        lab = cv2.cvtColor(ccc_image, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)
    else:
        assert (True)

    delta_L = L - 255.0
    delta_L = erosion(delta_L, square(3))
    d = np.sqrt(np.square(a) + np.square(b) + np.square(delta_L))
    med = np.median(d) / np.std(d)
    return {'image': d, 'score': med}


def shadow(ccc_image, islab=False):
    ''' White is 255,255,255 in BGR and 255,0,0 in Lab
     Not handling rgb passed in as tuple
     '''
    if (isinstance(ccc_image, tuple)):
        assert (islab)

    if isinstance(ccc_image, tuple) and islab is True:
        L, a, b = ccc_image
    elif islab is True:
        L, a, b = cv2.split(ccc_image)
    elif not islab:
        lab = cv2.cvtColor(ccc_image, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)
    else:
        assert (True)

    delta_L = L - 255.0
    delta_L = erosion(delta_L, square(3))
    d = np.sqrt(np.square(a) + np.square(b) + np.square(delta_L))
    med = np.median(d) / np.std(d)
    return {'image': d, 'score': med}



if __name__ == "__main__":
    if len(sys.argv) < 2:
        exit(1)
    print(sys.argv[1])
    if Path(sys.argv[1]).is_file():
        img = fetch_image_for_image_file(sys.argv[1])
        if img.ndim > 3:
            rgb = color.rgba2rgb(img)
        elif img.ndim == 3:
            rgb = img
        else:
            print('Not a color image')
            exit(1)

        Red = rgb[:, :, 0]
        Green = rgb[:, :, 1]
        Blue = rgb[:, :, 2]

        gg = geom_gray(rgb)
        mm = mean_gray(rgb)

        f, axs = plt.subplots(2, 4, figsize=(20, 10), frameon=False,
                              subplot_kw={'xticks': [], 'yticks': []})
        axs[0, 0].imshow(rgb)
        axs[0, 1].imshow(Red, cmap='gray')
        axs[0, 2].imshow(Green, cmap='gray')
        axs[0, 3].imshow(Blue, cmap='gray')
        axs[1, 0].imshow(gg, cmap='gray')
        axs[0, 1].imshow(mm, cmap='gray')

        plt.autoscale
        plt.show()
