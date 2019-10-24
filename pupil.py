import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from skimage import color, img_as_ubyte, img_as_float
from skimage.color import rgb2gray
from operator import itemgetter
from sobelplus import sobel_detect, gradient_all
from skimage.util import img_as_ubyte
from skimage.util import img_as_float
from circles import find_circle
from skimage.draw import circle_perimeter
from matplotlib.patches import Circle
import matplotlib.patches as patches
import math

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


def CenterOfIntensity(gray, invert=False):
    src = gray.copy()
    if invert:
        src = cv2.bitwise_not(gray)

    moments = cv2.moments(src)  # Calculate moments
    (cx, cy) = (-1.0, -1.0)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
        cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
    return (cx, cy), moments['m00']

    # threshold the image
    # otsu


def getMask(rc):
    ret, otsu = cv2.threshold(rc, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu_eroded = cv2.erode(otsu, None, iterations=2)
    otsu_cleansed = cv2.dilate(otsu_eroded, None, iterations=2)
    mask = createConvexMask(otsu_cleansed)
    return mask, otsu


def smooth(x, window_len):
    # smooth the data using a window with requested size.

    assert (x.ndim is 1)
    assert (x.size > window_len)
    if window_len < 3:
        return x

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    w = np.ones(window_len, 'd')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def pupil_range_by_reduce(roi, low_pass):
    colsums = cv2.reduce(roi, 0, cv2.REDUCE_MAX, dtype=cv2.CV_8U).flatten()
    roi_width = len(colsums)
    diffc = np.diff(colsums)
    if roi_width > (low_pass * 2):
        diffc = smooth(diffc, low_pass)
    medc = np.median(diffc)
    sortc = np.argwhere(diffc < medc)

    fsortc = (sortc).flatten()
    fsortd = np.diff(fsortc)
    esort = np.select([fsortd], [fsortd > 1])
    posts = []

    for idx, val in enumerate(esort):
        if idx == 0: posts.append(fsortc[idx])
        if idx == len(esort) - 1: posts.append(fsortc[idx])
        if val == 1:
            posts.append(fsortc[idx])
            posts.append(fsortc[idx + 1])
    runs = []
    for idx, post in enumerate(posts):
        if idx % 2 == 0:
            runs.append([post, posts[idx + 1], posts[idx + 1] - post])

    lengths = np.diff(posts)
    print(lengths)
    print(runs)

    def Sort(sub_li):

        # reverse = None (Sorts in Ascending order)
        # key is set to sort using second element of
        # sublist lambda has been used
        return (sorted(sub_li, key=lambda x: x[2], reverse=True))

    sruns = Sort(runs)
    est_pupil = sruns[0]
    return est_pupil

def pupil(img, dpath=None):
    src = img.copy()
    low_pass = 5
    src = cv2.GaussianBlur(src, (low_pass, low_pass), 0)
    tmp = create_new(src)
    dest = create_new(src)
    height, width, channels = img.shape
    image_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    image_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    Lc, Ac, Bc = cv2.split(image_lab)
    cm, wmass = CenterOfIntensity(Lc)
    print('Entire Cm ', cm)

    image_gray = color.rgb2gray(img_as_float(image_rgb))
    high_radi = height * 0.31
    low_radi = high_radi - 30
    print('radii range', (low_radi, high_radi))

    hough_radii = np.arange(low_radi, high_radi, 2)
    circle_zip, edges = find_circle(image_gray, hough_radii, 1)
    # note row column to x y to width height
    circles = []
    for center_y, center_x, radius in circle_zip:
        circles.append((center_x, center_y, radius))

    iris_circle = circles[0]
    crad = (center_x, center_y)
    center_y = iris_circle[1]
    center_x = iris_circle[0]
    print('Center ', (center_x, center_y))

    roi = Lc[int(center_y - radius + 16):int(center_y + radius - 16),
          int(center_x - radius + 16):int(center_x + radius - 16)]
    cmroi, roimass = CenterOfIntensity(roi, True)
    print('cmroi', cmroi)

    est_pupil = pupil_range_by_reduce(roi, low_pass)
    estimated_pupil_radius = est_pupil[2] / 2
    estimated_pupil_x_center = center_x + radius - (est_pupil[1] + est_pupil[0]) / 2

    # @note refactor to using rectangles
    ring = 0.25
    tl_x = center_x - radius
    tl_y = center_y - radius
    pwidth = radius * (1.0 + ring)
    pheight = radius * (1.0 + ring)
    proi_tl_x = tl_x + (1.0 - ring) * radius / 2
    proi_tl_y = tl_y + (1.0 - ring) * radius / 2
    print('cmroi_tl', (proi_tl_x, proi_tl_y))

    # reduce the roi by the difference of roi and roi cm.
    proi = Lc[int(proi_tl_y):int(proi_tl_y + pheight),
           int(proi_tl_y):int(proi_tl_y + pheight)]

    cmproi, pmass = CenterOfIntensity(proi, True)
    prad = np.sqrt((pmass / 256.0) / math.pi)
    print('cmproi', (cmproi, prad))

    # Preprocessing images
    src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    circle_color = 'red'
    box_color = 'blue'
    pupil_color = 'yellow'

    #    fig = plt.figure(figsize=((22, 13)))
    f1, ax1 = plt.subplots(1)
    ax1.set_aspect('equal')
    ax1.imshow(src_rgb)
    ax1.set_title("Pupil Results")


    print((center_x, center_y, radius))
    c = patches.Circle((center_x, center_y), radius, color=circle_color, linewidth=2, fill=False)
    ax1.add_patch(c)
    r = patches.Rectangle((center_x - radius, center_y - radius), radius * 2, radius * 2, color=box_color,
                          linewidth=3, fill=False)
    ax1.add_patch(r)
    p = patches.Circle((estimated_pupil_x_center, center_y), estimated_pupil_radius, color=pupil_color, linewidth=1,
                       fill=False)
    ax1.add_patch(p)
    lc = patches.Rectangle((proi_tl_x, proi_tl_y), pwidth, pheight, color='yellow', linewidth=1, fill=False)
    ax1.add_patch(lc)

    #    l = patches.Rectangle((cmroi[0] + tl_x, cmroi[1] + tl_y), 7, 7, color='green', linewidth=3, fill=False)
    #    ax1.add_patch(l)

    pp = patches.Circle((proi_tl_x + cmproi[0], proi_tl_y + cmproi[1]), prad, color=pupil_color,
                        linewidth=2,
                        fill=False)
    ax1.add_patch(pp)
    plt.show()

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    pupil(img)
