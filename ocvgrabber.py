import os
import numpy as np
import cv2
import sys
import math
from pathlib import Path
import pickle
from padChecker import padChecker
from matplotlib import pyplot as plt
import time
import argparse  # provide interface for calling this script
from coloralgo import CenterOfIntensity
from opencv_utils import drawString
from vp import geom_tools


def circleContainsPoint(center, radius_2, point):
    dx = center[0] - point[0]
    dy = center[1] - point[1]
    dd = dy * dy + dx * dx
    return dd < radius_2


def circleContainsLine(center, radius_2, line):
    x1, y1, x2, y2 = line
    if not circleContainsPoint(center, radius_2, (x1, y1)): return False
    if not circleContainsPoint(center, radius_2, (x2, y2)): return False
    return True


def circleContainsBoundingRect(center, radius_2, bb):
    if not circleContainsPoint(center, radius_2, (bb[0], bb[1])): return False
    if not circleContainsPoint(center, radius_2, (bb[0] + bb[2], bb[1])): return False
    if not circleContainsPoint(center, radius_2, (bb[0] + bb[2], bb[1] + bb[3])): return False
    if not circleContainsPoint(center, radius_2, (bb[0], bb[1] + bb[3])): return False
    return True


def draw_cross(img, center, color, d):
    cv2.line(img,
             (center[0] - d, center[1]), (center[0] + d, center[1]),
             color, 1, cv2.LINE_AA, 0)
    cv2.line(img,
             (center[0], center[1] - d), (center[0], center[1] + d),
             color, 1, cv2.LINE_AA, 0)


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
    min_line_length = 5  # min_line_length * diagonal
    max_line_length = 66  # max_line_length * diagonal
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
    line_lengths = [geom_tools.get_line_length(l[0]) for l in lines]
    line_angles = [geom_tools.get_line_angle(l[0]) for l in lines]

    return [l[0] for (i, l) in enumerate(lines)
            if max_line_length > line_lengths[i] > min_line_length and
            precisions[i] > min_precision]

global global_settings

def initialize_settings(frame_tl, capture_size):
    settings = {'frameTopLeft': frame_tl, 'active_center_norm': (0.5, 0.5), 'active_radius_norm': 0.4,
                'capture_size': capture_size}
    width = int(settings['capture_size'][0] + 0.5) - settings['frameTopLeft'][0] * 2
    height = int(settings['capture_size'][1] + 0.5) - settings['frameTopLeft'][1] * 2
    settings['frame_size'] = (width, height)
    settings['frameBottomRight'] = (frame_tl[0] + width, frame_tl[1] + height)
    settings['active_center'] = (int(width * settings['active_center_norm'][0]),
                                  int(height * settings['active_center_norm'][1]))

    settings['active_radius'] = int(height * settings['active_radius_norm'])
    settings['frame_count'] = 0
    return settings


def process_frame(frame, checker, settings_dict):
    settings = settings_dict
    settings['frame_count'] = settings['frame_count'] + 1
    tl = settings['frameTopLeft']
    br = settings['frameBottomRight']
    frame_roi = frame[tl[1]: br[1], tl[0]: br[0]]
    display = frame[tl[1]: br[1], tl[0]: br[0]]
    lab_image = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    # Split LAB channels
    gray, a, b = cv2.split(lab_image)
    gray = gray[tl[1]: br[1], tl[0]: br[0]]

    start_time = time.time()
    y, mask, seethrough, cnts = checker.check(frame_roi)

    radius_2 = settings['active_radius']
    radius_2 = radius_2 * radius_2

    bbs = []
    contours = []
    for cnt in cnts:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle
        bounding_rect = cv2.boundingRect(cnt)
        if circleContainsBoundingRect(settings['active_center'], radius_2, bounding_rect):
            bbs.append(bounding_rect)
            contours.append(cnt)
            cv2.drawContours(display, cnt, -1, (128, 128, 0), -1)

    # for bb in bbs:
    #     area = bb[2] * bb[3]
    #     min_radius = math.sqrt(area/math.pi)
    #     cx = int(bb[0] + bb[2] / 2)
    #     cy = int(bb[1] + bb[3] / 2)
    #     draw_cross(display, (cx, cy), (0, 0, 255), 10)
    #     cv2.circle(display, (cx, cy), int(min_radius + 0.5), (255, 0, 0), -1)

    check_time = time.time()
    raw_lines = lsd_lines(gray)
    lines = []
    for line in raw_lines:
        if circleContainsLine(settings['active_center'], radius_2, line):
            lines.append(line)

    lsd_time = time.time()
    lsd_time = lsd_time - check_time
    check_time = check_time - start_time
    print((check_time, lsd_time))

    for line in lines:
        angle = geom_tools.get_line_angle(line)
        vert = angle % 90
        horz = angle % 180
        x1, y1, x2, y2 = line
        cv2.line(display, (x1, y1), (x2, y2), (0, vert * 255 / 90., horz * 255 / 180.), 2)

    vals = checker.history()
    maxi = np.max(vals)
    x = np.arange(len(vals))
    vals = np.multiply(settings['frame_size'][1], vals)
    vals = np.subtract(settings['frame_size'][1], vals)
    pts = np.vstack((x, vals)).astype(np.int32).T
    cv2.polylines(display, [pts], isClosed=False, color=(255, 0, 0))

    drawString(frame, str(fcount))
    cv2.circle(display, settings['active_center'], settings['active_radius'], (0, 255, 0), 3)
    res = np.vstack((display, seethrough))
    return res


if __name__ == "__main__":

    # Get input parameters
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--source", required=False, default='camera', help="camera -- default is camera")
    ap.add_argument("-v", "--video", required=False, help="path to input video file")
    ap.add_argument("-c", "--cache", required=False, default='./projects/wiic',
                    help="path to cache directory relative to source code directory")
    args = vars(ap.parse_args())
    file_folder = os.path.dirname(os.path.realpath(__file__))
    is_file = args['source'] == 'file' and Path(args['video']).is_file()
    is_camera = not is_file
    cachePath = os.path.abspath(os.path.join(file_folder, args['cache']))
    file_name = args['video']
    checker = padChecker(cachePath)
    if is_file:
        cap = cv2.VideoCapture(file_name)  # Capture video from camera
    if is_camera:
        cap = cv2.VideoCapture(0)  # Capture video from camera

    # setup settings based on capture size and padding for black regions around
    settings = initialize_settings((10, 60),(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    ret, frame = cap.read()
    if ret == False: exit(1)
    fcount = 1
    while (cap.isOpened()):
        ret, frame = cap.read()
        res = process_frame(frame, checker, settings)
        cv2.imshow('frame', res)
        kk = cv2.waitKey(1) & 0xff
        Pause = kk == ord('c')
        if kk == ord('n') or Pause == False:
            continue
        else:
            if kk == ord('q'):  # Hit `q` to exit
                break
    else:
        exit(0)

    # Release everything if job is finished
    # if we were writing out.release()
    cap.release()
    cv2.destroyAllWindows()
    exit(0)
