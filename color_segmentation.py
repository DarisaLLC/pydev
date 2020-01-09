import os
import sys
import glob
import argparse
from sklearn.cluster import KMeans
from pathlib import Path
from skimage.feature import peak_local_max
from skimage import img_as_float, img_as_ubyte, img_as_bool, img_as_int
from skimage import measure

from scipy.spatial import distance as dist
from scipy import optimize
from scipy import ndimage

import numpy as np
import argparse
import cv2
from utils import centroid_histogram, plot_colors
from matplotlib import pyplot as plt

'''
args_colorspace hsv, lab, ycc, ycrcb, bgr
args_channels single character channel identifier or all
args_num_clusters = number of expected clusters




'''



def color_cluster_seg(image, args_colorspace, args_channels, args_num_clusters, args_plot=False):
    # Change image color space, if necessary.
    colorSpace = args_colorspace.lower()

    if colorSpace == 'hsv':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    elif colorSpace == 'ycrcb' or colorSpace == 'ycc':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    elif colorSpace == 'lab':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    else:
        colorSpace = 'bgr'  # set for file naming purposes

    # Keep only the selected channels for K-means clustering.
    if args_channels != 'all':
        channels = cv2.split(image)
        channelIndices = []
        for char in args_channels:
            channelIndices.append(int(char))
        image = image[:, :, channelIndices]
        if len(image.shape) == 2:
            image.reshape(image.shape[0], image.shape[1], 1)

    (width, height, n_channel) = image.shape

    # Flatten the 2D image array into an MxN feature vector, where M is the number of pixels and N is the dimension (number of channels).
    reshaped = image.reshape(image.shape[0] * image.shape[1], image.shape[2])

    # K-means clustering. Use Atleast Two two
    if args_num_clusters < 2:
        print('Warning: num-clusters < 2 invalid. Using num-clusters = 2')
    numClusters = max(2, args_num_clusters)

    # clustering method. Could use OpenCv. Skitlearn is a lot better
    kmeans = KMeans(n_clusters=numClusters, n_init=40, max_iter=500).fit(reshaped)

    # get lables
    pred_label = kmeans.labels_

    # Reshape result back into a 2D array, where each element represents the corresponding pixel's cluster index (0 to K - 1).
    clustering = np.reshape(np.array(pred_label, dtype=np.uint8), (image.shape[0], image.shape[1]))

    # Sort the cluster labels in order of the frequency with which they occur.
    sortedLabels = sorted([n for n in range(numClusters)], key=lambda x: -np.sum(clustering == x))


    # Initialize K-means grayscale image; set pixel colors based on clustering.
    kmeansImage = np.zeros(image.shape[:2], dtype=np.uint8)
    for i, label in enumerate(sortedLabels):
        kmeansImage[clustering == label] = int(255 / (numClusters - 1)) * i
        kImage = np.zeros(image.shape[:2], dtype=np.uint8)
        kImage[clustering == label] = 255
        filename = '/Users/arman/Pictures/ceye/slices/slice_' + str(i) + '.png'
        cv2.imwrite(filename, kImage)
        print(i)
        cv2.namedWindow('Slice', cv2.WINDOW_NORMAL)
        cv2.imshow('Slice', kImage)
        cv2.waitKey(0)
    ret, thresh = cv2.threshold(kmeansImage, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    if args_plot:
        hist = centroid_histogram(kmeans)
        bar = plot_colors(hist, kmeans.cluster_centers_)

        # show our color bart
        plt.figure()
        plt.axis("off")
        plt.imshow(bar)
        plt.show()

    return thresh, kmeansImage, ret


'''
def medial_axis_image(thresh):
 
    #convert an image from OpenCV to skimage
    thresh_sk = img_as_float(thresh)

    image_bw = img_as_bool((thresh_sk))

    image_medial_axis = medial_axis(image_bw)

    return image_medial_axis
'''


if __name__ == '__main__':
    if not Path(sys.argv[1]).is_file() or not Path(sys.argv[1]).exists():
        print(sys.argv[1] + '  Does not exist ')
    img = cv2.imread(sys.argv[1])
    rows, cols, channels = map(int, img.shape)


#    cols = cols // 2
#    rows = rows // 2


    #img = cv2.medianBlur(img, 5)

    thresh, kmimage, ret = color_cluster_seg(img, 'hsv', 'all', 8, True)
    print(thresh)

    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', img)
    cv2.namedWindow('colorSeg', cv2.WINDOW_NORMAL)
    cv2.imshow('colorSeg', kmimage)
    cv2.namedWindow('colorThreshold', cv2.WINDOW_NORMAL)
    cv2.imshow('colorThreshold', kmimage)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
