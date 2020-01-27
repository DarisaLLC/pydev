import numpy as np
import sys
import os
import math
import cv2 as cv
import datetime
import logging
from datetime import datetime
from pathlib import Path
from sympy import simplify, Polygon, convex_hull, Point2D, Line2D, Line
import itertools
from simple_kdtree import make_kd_tree, add_point, get_knn, get_nearest, PointContainer

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
                                 maxLevel=3,
                                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

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

    cv.fillConvexPoly(mask, poly, 255)
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

    cv.fillConvexPoly(mask, poly, 255)
    return mask


def _draw_str(dst, target, s, scale=1.0):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, scale, (0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, scale, (128, 128, 128), lineType=cv.LINE_AA)


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
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
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
    lsd = cv.createLineSegmentDetector(0) #cv.LSD_REFINE_STD, 0.8, 0.6, 2.0, 22.5, 0, 0.9, 1024)

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


def midPoint(line):
    p0, p1 = np.array([line[0], line[1]]), np.array([line[2], line[3]])
    return np.array((p0 + p1) / 2)


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


def quadrilateral_detection(lines, locations, directions, strengths, kdt, line_limits=(50, 100), parallel_thr=5.0):
    cands = []
    N = len(lines)
    parallels = []
    dim = 2

    def dist_sq(a, b, dim):
        return sum((a[i] - b[i]) ** 2 for i in range(dim))

    def dist_sq_dim(a, b):
        return dist_sq(a, b, dim)

    iterator = itertools.combinations(range(N), 2)
    for i, j in iterator:
        dD = angle_diff(directions[i], directions[j])
        if math.degrees(dD) > parallel_thr: continue
        mid_i = locations[i]
        mid_j = locations[j]
        midp = midPoint([mid_i[0], mid_i[1], mid_j[0], mid_j[1]])
        nknn = get_knn(kdt, midp, 8, dim, dist_sq_dim)
        distances = []
        lids = []
        for nn in nknn:
            distances.append(nn[0])
            lids.append(nn[1].name)
        ulids = uniques(lids)
        lulids = len(list(ulids))
        if lulids == 8: continue
        print(len(list(ulids)), ' Uniques')
        #
        # dis = np.linalg.norm(mid_i - mid_j)
        # if dis < line_limits[0] and dis < line_limits[1]:
        #     parallels.append([mid_i[0],mid_i[1],mid_j[0],mid_j[1]])

    return parallels


'''
 Lines are already been pruned to same angle 
 We are looking for quadrilaterals

 '''


def line_yhist(lines, frame_size):
    # create a y histogram of lines
    #   la = np.array(lines, dtype='float').view(np.recarray)
    # Remove singleton dimension
    la = lines[:, 0]

    radians = np.arctan2(la[:, 3] - la[:, 1], la[:, 2] - la[:, 0])
    nags = radians < 0
    radians[nags] += math.pi
    radians = np.remainder(radians, math.pi)

    for line in lines:
        p1, p2 = map(Point2D, [(line[0], line[1]), (line[2], line[3])])
        L1 = Line2D(p1, p2)
        angle_raw = xaxis.angle_between(L1).evalf()
        if angle_raw < 0:
            angle_raw += math.pi
        angle = angle_raw % math.pi
        print(('angle ', math.degrees(angle)))
        d = L1.length.evalf()
        dd = d * d
        y = int(line[1])
        yhist[y][0] += 1
        yhist[y][1] += d
        yhist[y][2] += dd

    return yhist


def lines2quadrilaterals(lines, image_size):
    def get_mid_point(line):
        return (line[0] + line[2]) / 2, (line[1] + line[3]) / 2

        # create a y histogram of lines

    yhist = np.zeros((image_size[1], 3), dtype=float)
    for line in lines:
        L1 = Line2D(line)
        yhist[line[0]] += 1
        yhist[line[3]] += 1

    cands = []
    for s1, s2 in combinations(lines, 2):
        # L1 = Line2D(s1)
        # L2 = Line2D(s2)
        # m1 = get_mid_point(s1)
        # m2 = get_mid_point(s2)

        points = [(s1[0], s1[1]), (s1[2], s1[3]), (s2[0], s2[1]), (s2[2], s2[3])]
        poly = convex_hull(*points)
        #        print(('check ', poly, poly.is_convex()))
        passit = True
        for other in lines:
            if same_line(s1, other) or same_line(s2, other): continue
            #            print(('against ', p1, p2))
            if poly.encloses_point((other[0], other[1])) or poly.encloses_point((other[2], other[3])):
                passit = False
                #                print(('bad ', poly))
                break
        # print(('good ', poly))
        if passit: cands.append(poly)
    return cands
