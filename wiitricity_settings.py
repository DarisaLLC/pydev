import numpy as np
import sys
import os
import math
import cv2
import datetime
import logging
from datetime import datetime
from pathlib import Path
from sympy import simplify, Polygon, convex_hull, Point2D, Line2D, Line
import itertools
from simple_kdtree import make_kd_tree, add_point, get_knn, get_nearest, PointContainer
import geometry as utils
from scipy.spatial import distance


'''
Initialization of common settings for WiiTricity image & videos

Image data location within a stored image or frame of video
frame_tl is the topleft corner of image data in the stored image or video frame
capture_size is the image size 

'''
video_rois = {'one': dict(row_low=53, row_high=350, column_low=6, column_high=708),
              'hd2': dict(row_low=0, row_high=759, column_low=0, column_high=1919),
              'hd': dict(row_low=0, row_high=519, column_low=0, column_high=1279)}


def initialize_settings_from_video_roi(video_roi):
    frame_tl = (video_roi['column_low'], video_roi['row_low'])
    capture_size = (video_roi['column_high'] - video_roi['column_low'], video_roi['row_high'] - video_roi['row_low'])
    return initialize_settings(frame_tl, capture_size)


def initialize_settings(frame_tl, capture_size):
    settings = {'frameTopLeft': frame_tl, 'active_center_norm': (0.5, 0.5), 'active_radius_norm': 0.4,
                'capture_size': capture_size, 'cache': '.'}

    width = int(settings['capture_size'][0] + 0.5) - settings['frameTopLeft'][0] * 2
    height = int(settings['capture_size'][1] + 0.5) - settings['frameTopLeft'][1] * 2
    settings['frame_size'] = (width, height)
    settings['frameBottomRight'] = (frame_tl[0] + width, frame_tl[1] + height)
    settings['active_center'] = (int(width * settings['active_center_norm'][0]),
                                 int(height * settings['active_center_norm'][1]))

    settings['active_radius'] = int(height * settings['active_radius_norm'])
    settings['frame_count'] = 0
    settings['ppMedianBlur'] = 7
    # Parameters for lucas kanade optical flow
    settings['lk_params'] = dict(winSize=(15, 15),
                                 maxLevel=4,
                                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # params for ShiTomasi corner detection
    settings['feature_params'] = dict(maxCorners=500,
                                      qualityLevel=0.3,
                                      minDistance=7,
                                      blockSize=7)
    #                          useHarrisDetector=False,
    #                          k=0.04)
    settings['max_distance'] = 5
    settings['min_features'] = 300
    settings['mask_diagonal_ratio'] = 2.5

    return settings


'''
Create a mask like this. 
diagonal ratio is  ration of ... to ...-----...

 ...------...
   /      \
  /        \
  ----------

'''


def region_of_interest(img, diagonal_ratio=2.5):
    rows, cols, channels = img.shape
    mask = np.zeros((rows, cols), dtype=np.uint8)
    left = int(cols / diagonal_ratio)
    right = int(cols - left)

    poly = np.array([[
        (left, 50), (right, 50), (cols - 1, rows - 1), (0, rows - 1)]], np.int32)

    cv2.fillConvexPoly(mask, poly, 255)
    return mask


'''
Create a mask like this. 
Higher portion

  ----------
  ----------

'''


def vertical(img, top_portion=0.667):
    rows, cols, channels = img.shape
    mask = np.zeros((rows, cols), dtype=np.uint8)
    top = int(rows * top_portion)

    poly = np.array([[
        (0, top), (cols - 1, top), (cols - 1, rows - 1), (0, rows - 1)]], np.int32)

    cv2.fillConvexPoly(mask, poly, 255)
    return mask


def _draw_str(dst, target, s, scale=1.0):
    x, y = target
    cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, scale, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, scale, (128, 128, 128), lineType=cv2.LINE_AA)


def create_logging_directory(output_path):
    client_output = output_path and Path(output_path).exists() and Path(output_path).is_dir()
    if not client_output and Path(output_path).parent.exists:
        os.mkdir(output_path)
    return Path(output_path).exists


# create logger with 'wiirunner'
def get_logger():
    # create a logger and setup console logger
    logger = logging.getLogger('wiirunner')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # create file logger if we can create the logging directory in logs sub directory under code
    logging_folder = os.path.dirname(os.path.realpath(__file__)) + '/logs'
    dir_good = create_logging_directory(logging_folder)
    if dir_good:
        logfilepath = logging_folder + '/wiirunner' + datetime.now().strftime("-%d-%m-%Y_%I-%M-%S_%p") + '.log'
        # create file handler which logs even debug messages
        fh = logging.FileHandler(logfilepath)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def bgrFromHue(degrees):
    hsv = np.zeros((1, 1, 3), np.uint8)
    hsv[0, 0, 0] = ((degrees % 180) * 256) / 180.0
    hsv[0, 0, 1] = 255
    hsv[0, 0, 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    tp = tuple([int(x) for x in bgr[0, 0, :]])
    return tp


def get_line_angle(line):
    x1, y1, x2, y2 = np.array(line, dtype=np.float64)
    radians = np.arctan2(y2 - y1, x2 - x1)
    return radians


def circular_mean(weights, angles):
    x = y = 0.
    for angle, weight in zip(angles, weights):
        x += math.cos(math.radians(angle)) * weight
        y += math.sin(math.radians(angle)) * weight

    mean = math.degrees(math.atan2(y, x))
    return mean


'''
mask region is a rectangle in tl image system
represented by tl and br coords

'''


def compute_lines(image, mask_region, length_limit):
    # Create LSD detector with default parameters
    '''
    cv2.LSD_REFINE_STD ,0.97, 0.6, 0.8, 40, 0, 0.90, 1024
    '''
    lsd = cv2.createLineSegmentDetector(0) #cv2.LSD_REFINE_STD, 0.8, 0.6, 2.0, 22.5, 0, 0.9, 1024)

    # Detect lines in the image
    # Returns a NumPy array of type N x 1 x 4 of float32
    # such that the 4 numbers in the last dimension are (x1, y1, x2, y2)
    # These denote the start and end positions of a line
    lines = lsd.detect(image)[0]

    # Remove singleton dimension
    lines = lines[:, 0]

    # Filter out the lines whose length is lower than the threshold
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    lengths = np.sqrt(dx * dx + dy * dy)
    mask = lengths > length_limit[0]
    lines = lines[mask]
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    lengths = np.sqrt(dx * dx + dy * dy)
    mask = lengths < length_limit[1]
    lines = lines[mask]

    locations = []
    strengths = []
    tl, br = mask_region
    new_lines = []
    new_directions = []
    points = []

    for line in lines:
        p0, p1 = np.array([line[0], line[1]]), np.array([line[2], line[3]])
        midp = (p0 + p1) / 2
        within = midp[0] >= tl[0] and midp[0] < br[0] and midp[1] >= tl[1] and midp[1] < br[1]
        if not within: continue
        locations.append((p0 + p1) / 2)
        strengths.append(np.linalg.norm(p1 - p0))
        new_lines.append(line)
        points.append(PointContainer(p0, name=len(points)))
        points.append(PointContainer(p1, name=len(points)))

    kdt = make_kd_tree(points, 2)
    # convert to numpy arrays and normalize
    locations = np.array(locations)
    strengths = np.array(strengths)
    new_lines = np.array(new_lines)
    directions = np.arctan2(new_lines[:, 3] - new_lines[:, 1], new_lines[:, 2] - new_lines[:, 0])



    return (new_lines, locations, directions, strengths, kdt)


def angle_diff(angle1: float, angle2: float) -> float:
    """Computes the angle difference between the input arguments.

    .. note:: The resulting angle difference is in [0, pi * 0.5]

    :param angle1: First angle expressed in radiants.
    :param angle2: Second angle expressed in radiants.
    :return: Angle difference between the input parameters. This angle represents the smallest positive angle between the input parameters.
    """
    d_angle = np.abs(angle1 - angle2)
    d_angle = d_angle % np.pi
    if d_angle > np.pi * 0.5:
        d_angle -= np.pi
    return np.abs(d_angle)


def get_mid_point(line):
    return np.array([(line[0] + line[2]) / 2, (line[1] + line[3]) / 2])

def get_distance(pt1,pt2):
    diff = pt1 - pt2

from collections.abc import Hashable


def uniques(iterable):
    seen_hashable = set()
    seen_unhashable = []
    for item in iterable:
        if isinstance(item, Hashable):
            if item not in seen_hashable:
                yield item
                seen_hashable.add(item)
        else:
            if item not in seen_unhashable:
                yield item
                seen_unhashable.append(item)


def quasi_quadrilateral_detection(lines, directions, angle_thr=math.pi/10, ortho_thr=math.pi/4.5, dis_thr=200):
    indexes = []
    mids = []
    N = len(lines)
    quads = []
    #order = utils.sort_segments(lines)

    iterator = itertools.combinations(range(N), 2)
    for i, j in iterator:
        Li = lines[i]
        Lj = lines[j]
        p0, p1 = np.array([Li[0], Li[1]]), np.array([Li[2], Li[3]])
        p2, p3 = np.array([Lj[0], Lj[1]]), np.array([Lj[2], Lj[3]])
        Ai = utils.angle_x(p0,p1)
        Aj = utils.angle_x(p2,p3)
        dt = utils.angle_diff(Ai,Aj)
        if dt > (angle_thr): continue

        pts = []
        pts.append(p0)
        pts.append(p1)
        pts.append(p2)
        pts.append(p3)
        # check overlap
        normal = np.array([0, 1])

        # Project the segment centers along the normal defined by the mean angle.
        projected_ends = np.array([np.dot(endp, normal) for endp in pts])
        d1 = np.maximum(projected_ends[0], projected_ends[1]) - np.minimum(projected_ends[0], projected_ends[1])
        d2 = np.maximum(projected_ends[2], projected_ends[3]) - np.minimum(projected_ends[2], projected_ends[3])
        order = np.argsort(projected_ends)
        em = order[0]
        en = order[3]
        pd = np.maximum(projected_ends[em], projected_ends[en]) - np.minimum(projected_ends[em], projected_ends[en])
        overlap = pd / (d1 + d2)
        if overlap > 1.1: continue
        mLi = get_mid_point(Li)
        mLj = get_mid_point(Lj)
        dist = distance.euclidean(mLi,mLj)
        check = dist < dis_thr
        if not check: continue
        qa = utils.area(np.array(pts))
        print('qa-area:', qa)

        indexes.append((i,j))
        mids.append([mLi,mLj])
        quads.append(np.array(pts))

    return indexes, mids, quads


