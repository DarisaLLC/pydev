import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from sobelplus import sobel_detect
from skimage.util import img_as_ubyte
from skimage.util import img_as_float

MASK_BORDER_SIZE = 7

SE_K_SIZE = 3


def choose(x):
    return {
        's': 1,
        'r': 2,
        'b': 3,
    }[x]


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


def CenterOfIntensity(gray):
    moments = cv2.moments(gray)  # Calculate moments
    (cx, cy) = (-1.0, -1.0)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
        cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
    return (cx, cy)


def pupil(img, sp, sr, dpath=None):
    src = img.copy()
    tmp = create_new(src)
    dest = create_new(src)
    i = 0
    while (i <= sr):
        src = cv2.cvtColor(src, cv2.COLOR_BGR2LAB, tmp)
        cv2.pyrMeanShiftFiltering(tmp, sp, i, dest, 2)
        dest = cv2.cvtColor(dest, cv2.COLOR_Lab2BGR, dest)
        src = dest
        i = i + 2

    if not (dpath is None):
        cv2.imwrite(dpath + 'mshift' + '.png', dest)
    bc, gc, rc = cv2.split(dest)
    cv2.medianBlur(rc, 7, rc)
    if not (dpath is None):
        cv2.imwrite(dpath + 'red' + '.png', rc)
    # threshold the image
    # otsu
    ret, otsu = cv2.threshold(rc, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu_eroded = cv2.erode(otsu, None, iterations=2)
    otsu_cleansed = cv2.dilate(otsu_eroded, None, iterations=2)

    mask = createConvexMask(otsu_cleansed)
    cx, cy = CenterOfIntensity(mask)
    print('(%d,%d)', (cx, cy))

    # remove any blobs that are on the edge

    # Preprocessing images
    src_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dest_rgb = cv2.cvtColor(dest, cv2.COLOR_BGR2RGB)
    f1, a1 = plt.subplots(3, 2, figsize=(18, 10))
    a1[0, 0].imshow(src_rgb), a1[0, 0].set_title("Original Image")
    a1[1, 0].imshow(dest_rgb), a1[1, 0].set_title("MeanShift")
    a1[0, 1].imshow(rc, cmap=plt.cm.gray), a1[0, 1].set_title("Red Channel")
    a1[1, 1].imshow(otsu, cmap=plt.cm.gray), a1[1, 1].set_title("Mask")
    a1[2, 1].imshow(mask, cmap=plt.cm.gray), a1[1, 1].set_title("Mask")
    f1.tight_layout()
    plt.show()


# # Watershed Segmentation
# f2, a2 = plt.subplots(2, 2, figsize=(18, 10))
# a2[0, 0].imshow(dtwImage), a2[0, 0].set_title("Distance Transform Weighted Image")
# a2[0, 1].imshow(markers), a2[0, 1].set_title("Markers for Wateshed")
# a2[1, 0].imshow(labels), a2[1, 0].set_title("Watershed Operation")
# a2[1, 1].imshow(segmented_image), a2[1, 1].set_title("Final Segmentation")
# f2.tight_layout()


if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    sp = int(sys.argv[2])
    sr = int(sys.argv[3])
    pupil(img, sp, sr)
