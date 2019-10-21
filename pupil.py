import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from skimage import color, img_as_ubyte, img_as_float
from skimage.color import rgb2gray

from sobelplus import sobel_detect, gradient_all
from skimage.util import img_as_ubyte
from skimage.util import img_as_float
from circles import find_circle
from skimage.draw import circle_perimeter
from matplotlib.patches import Circle
import matplotlib.patches as patches


def wait():
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_new(img):
    height, width, channels = img.shape
    return np.zeros((height, width, 3), np.uint8)


def convert_lab2rgb(lab_tuple):
    merged = cv2.merge(lab_tuple)
    return cv2.cvtColor(merged.astype("uint8"), cv2.COLOR_LAB2RGB)


def createConvexMask(image):
    """
    creates a convex mask (binary image) of foreground object of a binary input image
    """
    contours = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
    hull_image = np.zeros_like(image.copy(), dtype=np.uint8)
    height, width = hull_image.shape
    if len(contours) == 0:
        return (None)

    for (i, c) in enumerate(contours):
        # compute the area of the contour along with the bounding box
        # to compute the aspect ratio
        area = cv2.contourArea(c)
        (x, y, w, h) = cv2.boundingRect(c)
        if x == 0 or (x + w) == width or y == 0 or (y + h) == height: continue
        hull = cv2.convexHull(c)
        cv2.drawContours(hull_image, [hull], 0, (255), -1)
    return (hull_image)


def check_specular_reflection(otsu):
    # look for black
    inv = cv2.bitwise_not(otsu)
    return inv, None


def CenterOfIntensity(gray):
    moments = cv2.moments(gray)  # Calculate moments
    (cx, cy) = (-1.0, -1.0)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
        cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
    return (cx, cy)

    # threshold the image
    # otsu


def getMask(rc):
    ret, otsu = cv2.threshold(rc, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu_eroded = cv2.erode(otsu, None, iterations=2)
    otsu_cleansed = cv2.dilate(otsu_eroded, None, iterations=2)
    mask = createConvexMask(otsu_cleansed)
    return mask, otsu


def pupil(img, dpath=None):
    src = img.copy()
    tmp = create_new(src)
    dest = create_new(src)
    height, width, channels = img.shape
    image_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    image_gray = color.rgb2gray(img_as_float(image_rgb))
    high_radi = height * 0.25
    low_radi = high_radi - 5
    hough_radii = np.arange(low_radi, high_radi, 2)
    circle_zip, edges = find_circle(image_gray, hough_radii, 1)
    # iris_candidates = list(circle_zip)

    #
    # rinc, reflects = check_specular_reflection(rc)
    # cx, cy = CenterOfIntensity(mask)
    # print((cx, cy))
    # inv_mask = cv2.bitwise_not(mask)
    # cx, cy = CenterOfIntensity(inv_mask)
    # print((cx, cy))
    #
    # mask3 = cv2.resize(mask, img.shape[1::-1])
    # mask3 = cv2.merge([mask3,mask3,mask3])
    # iris = cv2.bitwise_and(img, mask3)
    #
    # hist_item = cv2.calcHist([img], [0], mask, [256], [0, 255])
    #
    #

    # remove any blobs that are on the edge

    # Preprocessing images
    src_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    circle_color = 'red'
    f1, a1 = plt.subplots(1)
    a1.set_aspect('equal')
    a1.imshow(src_rgb)
    a1.set_title("Pupil Results")

    for center_y, center_x, radius in circle_zip:
        c = patches.Circle((center_x, center_y), radius, color=circle_color, linewidth=2, fill=False)
        a1.add_patch(c)

    f1.tight_layout()
    plt.show()

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    pupil(img)
