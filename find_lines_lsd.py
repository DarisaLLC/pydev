import os
import numpy as np
import cv2
import sys
import math
from pathlib import Path
from matplotlib import pyplot as plt
import time
import argparse  # provide interface for calling this script
import utils
import opencv_utils
from ransac_vanishing_point import ransac_vanishing_point_detection
from circles import find_circle
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform
from skimage.draw import ellipse

def lsd_lines(source_image, min_line_length=0.0175, max_line_length=0.1, min_precision=0):
    """LSD algorithm for line detection.
    Args:
        source_image: An OpenCV Image.
        min_line_length: Minimum line size. Specified as a percentage of the
            source image diagonal (0-1).
        max_line_length: Maximum line size. Specified as a percentage of the
            source image diagonal (0-1).
        min_precision: Minimum precision of detections.
    Returns:
        Array of line endpoints tuples (x1, y1, x2, y2).
    """
    height, width = source_image.shape[:2]
    diagonal = math.sqrt(height ** 2 + width ** 2)
    min_line_length = 10  # min_line_length * diagonal
    max_line_length = 2660  # max_line_length * diagonal
    """
      createLineSegmentDetector([, _refine[, _scale[, _sigma_scale[, _quant[, _ang_th[, _log_eps[, _density_th[, _n_bins]]]]]]]]) -> retval
      .   @brief Creates a smart pointer to a LineSegmentDetector object and initializes it.
      .   
      .   The LineSegmentDetector algorithm is defined using the standard values. Only advanced users may want
      .   to edit those, as to tailor it for their own application.
      .   
      .   @param _refine The way found lines will be refined, see #LineSegmentDetectorModes
      .   @param _scale The scale of the image that will be used to find the lines. Range (0..1].
      .   @param _sigma_scale Sigma for Gaussian filter. It is computed as sigma = _sigma_scale/_scale.
      .   @param _quant Bound to the quantization error on the gradient norm.
      .   @param _ang_th Gradient angle tolerance in degrees.
      .   @param _log_eps Detection threshold: -log10(NFA) \> log_eps. Used only when advance refinement
      .   is chosen.
      .   @param _density_th Minimal density of aligned region points in the enclosing rectangle.
      .   @param _n_bins Number of bins in pseudo-ordering of gradient modulus.
      """
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)

    lines, rect_widths, precisions, false_alarms = detector.detect(source_image)
    line_lengths = [utils.get_line_length(l[0]) for l in lines]
    line_angles = [utils.get_line_angle(l[0]) for l in lines]

    return [l[0] for (i, l) in enumerate(lines)
            if max_line_length > line_lengths[i] > min_line_length and
            precisions[i] > min_precision]



def calcVanishingPoint(image):
    lines = lsd(image)
    points = lines[:, 2:3]
    normals = lines[:, 2:4] - lines[:, :2]
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-4)
    normals = np.stack([normals[:, 1], -normals[:, 0]], axis=1)
    normalPointDot = (normals * points).sum(1)

    if lines.shape[0] == 2:
        VP = np.linalg.solve(normals, normalPointDot)
    else:
        VP = np.linalg.lstsq(normals, normalPointDot)[0]
        pass

    return VP


if __name__ == '__main__':
    import sys


    def bgrFromHue(ang):
        hsv = np.zeros((1, 1, 3), np.uint8)
        hsv[0, 0, 0] = ((math.degrees(ang) % 360) * 256) / 360.0
        hsv[0, 0, 1] = ((math.degrees(ang) % 90) * 256) / 90.0
        hsv[0, 0, 2] = ((math.degrees(ang) % 45) * 256) / 45.0
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return bgr[0, 0, 0], bgr[0, 0, 1], bgr[0, 0, 2]


    display = None
    if len(sys.argv) < 2 or (not Path(sys.argv[1]).is_file() or not Path(sys.argv[1]).exists()):
        print(' File Does not exist or found ')
        sys.exit(1)


    lab_tuple = opencv_utils.load_reduce_convert(sys.argv[1], 1)
    display = opencv_utils.convert_lab2bgr(lab_tuple)
    dshape = display.shape
    print(dshape)

    coords = corner_peaks(corner_harris(lab_tuple[0]), min_distance=32)
    coords_subpix = corner_subpix(lab_tuple[0], coords, window_size=32)

    fig, ax = plt.subplots()
    ax.imshow(lab_tuple[0], cmap=plt.cm.gray)
    ax.plot(coords[:, 1], coords[:, 0], color='cyan', marker='o',
            linestyle='None', markersize=6)
    ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
    ax.axis((0, dshape[1], dshape[0], 0))
    plt.show()

    lines = lsd_lines(lab_tuple[0])

    vp = ransac_vanishing_point_detection(lines, distance=10)

    cv2.circle(display, vp, 10, (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
    cv2.circle(display, vp, 30, (0.0, 1.0, 1.0), 0, 3)
    print(vp)

    for line in lines:
        angle = utils.get_line_angle(line)
        x1, y1, x2, y2 = line
        b, g, r = bgrFromHue(angle)
        cv2.line(display, (x1, y1), (x2, y2), (b * 1.0, g * 1.0, r * 1.0, 0.5), 2)

    hough_radii = np.arange(20, 50, 3)
    circle_zip, edges = find_circle(lab_tuple[0], hough_radii, 1)
    # note row column to x y to width height
    circles = []
    for center_y, center_x, radius in circle_zip:
        circles.append((center_x, center_y, radius))
        cv2.circle(display, (center_x, center_y), radius, (0.0, 1.0, 1.0), 0, 3)

    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow("Display", display)
    key = cv2.waitKey(0) & 0xFF
