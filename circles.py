import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

from skimage import img_as_ubyte
from skimage import measure
from skimage import filters
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage import color, img_as_ubyte, img_as_float
from skimage.color import rgb2gray

# Construct some test data
#x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
#r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))

def find_circle(image_gray, radi_range, num_peaks):

    edges = canny(image_gray, sigma=4.0,low_threshold=0.01, high_threshold=0.15)

    # Detect two radii
    hough_radii = radi_range
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent 3 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=num_peaks)
    return zip(cy,cx,radii),edges

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    low_radi = int(sys.argv[2])
    high_radi = int(sys.argv[3])
    step_radi = int(sys.argv[4])
    num_peaks = int(sys.argv[5])

    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_gray = color.rgb2gray(img_as_float(image_rgb))

    hough_radii = np.arange(low_radi, high_radi, step_radi)

    circle_zip,edges = find_circle(image_gray,hough_radii, num_peaks)

    # Draw them
    gray_rgb = cv2.cvtColor(img_as_ubyte(edges),cv2.COLOR_GRAY2RGB)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    for center_y, center_x, radius in circle_zip:
        circy, circx = circle_perimeter(center_y, center_x, radius,
                                        shape=image_gray.shape)
        gray_rgb[circy, circx] = (220, 20, 20)

    ax.imshow(img_as_float(gray_rgb)) # cmap=plt.cm.gray)
    plt.show()

