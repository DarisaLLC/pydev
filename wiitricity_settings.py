import numpy as np
import sys
import os
import math
import cv2
import datetime
import logging
from datetime import datetime
from pathlib import Path
import itertools
import geometry as utils
from itertools import combinations
from scipy.spatial.distance import cosine as cosine_scipy
from sklearn.neighbors import KDTree
from math import sqrt


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
    settings['use_channel'] = 'hsv' # supports hsv as well
    width = int(settings['capture_size'][0] + 0.5) - settings['frameTopLeft'][0] * 2
    height = int(settings['capture_size'][1] + 0.5) - settings['frameTopLeft'][1] * 2
    settings['frame_size'] = (width, height)
    settings['frameBottomRight'] = (frame_tl[0] + width, frame_tl[1] + height)
    settings['active_center'] = (int(width * settings['active_center_norm'][0]),
                                 int(height * settings['active_center_norm'][1]))

    settings['active_radius'] = int(height * settings['active_radius_norm'])
    settings['frame_count'] = 0
    settings['reduction'] = 2
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
    settings['vertical_horizon_norm'] = 0.15
    settings['write_frames_path'] = None
    # Other Choices 'hue' or 'gray'
    settings['display_source'] = 'native_color'
    settings['expected_minimum_size'] = [18,30]
    settings['display_frame_delay_seconds'] = 100 # -1 means dont delay
    settings['display_click_after_frame'] = False
    settings['restrict_to_view_angle'] = True
    
    
    ## Synthesize, runs input video but instead of captured frames it synthesizes a moving / rotating rectangle
    
    settings['synthesize_test'] = False
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




def _draw_str(dst, target, s, scale):
    x, y = target
    cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_COMPLEX_SMALL, scale, (255,255,255), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, scale, (64,64,64), lineType=cv2.LINE_AA)


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


def point_2d_norm(point_0):
    """Returns the norm of a 2d point.
    :param point_0:
    :return:
    """
    return (point_0[0] ** 2 + point_0[1] ** 2) ** 0.5


def point_is_near_point(point_0, point_1, error=0.001):
    """Return true iff the two points are near each other.
    :param point_0:
    :param point_1:
    :param error:
    :return:
    """
    return point_2d_norm(point_0 - point_1) < error


def point_line_projection(point, line_start, line_end):
    """Return the projection of the point on the line.
    :param point:
    :param line_start:
    :param line_end:
    :return:
    """
    # Determine the projection of the point to the line
    v = line_end - line_start
    w = point - line_start
    return line_start + np.dot(v, w) * v / (point_2d_norm(v) ** 2)


def point_line_distance_and_projection(point, line_start, line_end):
    """Return the distance from the point to the projection of the point on the line.
    :param point:
    :param line_start:
    :param line_end:
    :return:
    """
    # Get the projection of the point
    projection = point_line_projection(point, line_start, line_end)
    # Return the distance between the projection and the point
    return point_2d_norm(projection - point), projection


def point_is_on_line(point, line_start, line_end, error=0.001):
    """Return true iff the point intersects the line.
    :param point:
    :param line_start:
    :param line_end:
    :param error:
    :return:
    """
    # We take the wedge- and dot product to determine if the point intersects the line
    v = line_end - point
    w = point - line_start
    return abs(np.linalg.det(np.column_stack((v, w)))) < error and np.dot(v, w) >= 0


def point_projection_is_on_line(point, line_start, line_end, error=0.001):
    """Return true iff the point projection intersects the line.
    :param point:
    :param line_start:
    :param line_end:
    :param error:
    :return:
    """
    # Determine the projection of the point to the line
    return point_is_on_line(point_line_projection(point, line_start, line_end), line_start, line_end, error)


def is_collinear(point_0, point_1, point_2, error=0.001):
    """Return true iff the three points are collinear.
    :param point_0:
    :param point_1:
    :param point_2:
    :param error:
    :return:
    """
    # We take the wedge product to determine if the three points are collinear
    v = point_2 - point_1
    w = point_0 - point_1
    return np.linalg.det(np.column_stack((v, w))) < error


def line_to_unit_interval(point, line_start, line_end, error=0.001):
    """Put a point on the unit interval of a line.
    :param point:
    :param line_start:
    :param line_end:
    :param error:
    :return:
    """
    if point_is_on_line(point, line_start, line_end, error):
        return point_2d_norm(point - line_start) / point_2d_norm(line_end - line_start)
    else:
        return None


def line_embedding(unit, line_start, line_end):
    """Embed a unit on to the line.
    :param unit:
    :param line_start:
    :param line_end:
    :return:
    """
    if 0.0 <= unit <= 1:
        return (1 - unit) * line_start + unit * line_end
    else:
        return None



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
If we are doing our own line detection
'''
def compute_lines(image, expected_orientation, length_limit, vertical_horizon, dlogger, dsettings):
    # Create LSD detector with default parameters
    '''
    cv2.LSD_REFINE_STD ,0.97, 0.6, 0.8, 40, 0, 0.90, 1024
    @todo Only run on below Horizon
    '''
    lsd = cv2.createLineSegmentDetector(0) #cv2.LSD_REFINE_STD, 0.8, 0.6, 2.0, 22.5, 0, 0.9, 1024)

    # Detect lines in the image
    # Returns a NumPy array of type N x 1 x 4 of float32
    # such that the 4 numbers in the last dimension are (x1, y1, x2, y2)
    # These denote the start and end positions of a line
    lines = lsd.detect(image)[0]
    # Remove singleton dimension
    lines = lines[:, 0]

    return filter_lines(lines, expected_orientation, length_limit, vertical_horizon, dlogger, dsettings)


def merge_lines (lines, max_merging_angle = np.pi * 180 / 5, max_endpoint_distance = 1):
    merged = []
    merged_segments = []
    for i, segment_i in enumerate(lines):
        if i in merged:
            continue
        collinears = [i]
        for j in range(i + 1, len(lines)):
            segment_j = lines[j]
            if utils.segments_collinear(segment_i, segment_j, max_angle = max_merging_angle,
                                        max_endpoint_distance = max_endpoint_distance):
                collinears.append(j)
        
        merged_segment = utils.merge_segments(lines[collinears])
        merged_segment = [int(m) for m in merged_segment]
        merged_segments.append(merged_segment)
        
        for index in collinears:
            if index not in merged:
                merged.append(index)

    return merged, np.array(merged_segments)
    


'''
Line Filtering 
mask region is a rectangle in tl image system
represented by tl and br coords

'''
def filter_lines(lines, expected_orientation, expected_size, vertical_horizon, dlogger, dsettings):
    
    N = len(lines)
    dlogger.info('Initial Line Count: ' + str(N))
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    xc = (lines[:, 2] + lines[:, 0]) / 2.0
    yc = (lines[:, 3] + lines[:, 1]) / 2.0
    
    # setup region for line processing
    mask = yc > vertical_horizon
    lines = lines[mask]
    dlogger.info('Post Horizon filter Line Count: ' + str(len(lines)))
    
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    xc = (lines[:, 2] + lines[:, 0]) / 2.0
    yc = (lines[:, 3] + lines[:, 1]) / 2.0
    
    directions = np.mod((np.arctan2(dy, dx) + np.pi), np.pi)
    lengths = np.sqrt(dx * dx + dy * dy)
    if dsettings['restrict_to_view_angle']:
        mask = np.mod(directions, expected_orientation) < 0.01
        lines = lines[mask]
        dx = dx[mask]
        dy = dy[mask]
        xc = xc[mask]
        yc = yc[mask]
        directions = directions[mask]
        lengths = lengths[mask]
        dlogger.info('Post Movement Direction filter Line Count: ' + str(len(lines)))
    
    N = len(lines)
#    out = np.zeros((N,N), dtype='float')
    iterator = itertools.combinations(range(N), 2)
    rects = []
    cands = []
    
    # Check on size
    # Check on parallelism cosine distance -> 1
    # Check on overlapness
    # Check on distance between centers
    # Check on whether mid to mid is prependicular to either line
    rng = np.random.RandomState(0)
    X = rng.random_sample((2,2))
    angle_thr = np.pi * 180 / 5
    
    for i, j in iterator:

        ## 3 lines: i, j and mid_i to mid_j
        ## first check if i and j are close to parallel
        # form mid_i_2_mid_j
        # check if mid_i_2_mid_j and either i or j are prependicular to each other
        ia = get_line_angle(lines[i])
        ja = get_line_angle(lines[j])
        if math.fabs(ia - ja) > angle_thr: continue
        m2m = (xc[i],yc[i],xc[j],yc[j])
        ma = get_line_angle(m2m)
        pp = ma - ia
        if pp < 0 : pp = pp + np.pi
        if math.fabs(pp - np.pi/2) > angle_thr: continue
        
        wide, short = i, j
        if lengths[wide] < lengths[short]:
            wide, short = j, i
        cands.append((wide,short))
        
        (xx1, yy1, xx2, yy2) = lines[wide]
        p1 = np.array((xx1, yy1))
        p2 = np.array((xx2, yy2))
        (x1, y1, x2, y2) = lines[short]
        p3 = np.array((x1, y1))
        p4 = np.array((x2, y2))
        d1,p13 = point_line_distance_and_projection(p3, p1, p2)
        d2,p14 = point_line_distance_and_projection(p4, p1, p2)
        
#        not_overlaps_in_x = np.maximum(p13[0], p14[0]) > np.minimum(p1[0],p2[0]) or np.maximum(p1[0], p2[0]) > np.minimum(p13[0],p14[0])
#        not_overlaps_in_y = np.maximum(p13[1], p14[1]) > np.minimum(p1[1],p2[1]) or np.maximum(p1[1], p2[1]) > np.minimum(p13[1],p14[1])
#        if not_overlaps_in_x or not_overlaps_in_y: continue
        
        points = [p1,p2,p3,p4]
        points = np.asarray(points, dtype=np.float32)
        rr = cv2.minAreaRect(points)
        (x, y), (width, height), angle = rr
        aspect_ratio = min(width, height) / max(width, height)
        if aspect_ratio < 0.33 : continue
        if angle < 0: angle = angle + 360.0
        slant = angle % (expected_orientation * 180 / np.pi)
        if slant > 5: continue
        

        
        rects.append(rr)

        
    return (rects, lines, directions, xc, yc, None)


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


def quasi_quadrilateral_detection(lines, directions, angle_thr=math.pi/10, ortho_thr=math.pi/4.5, dis_thr=300):
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
        if dt > (angle_thr*2): continue
        dt = utils.angle_diff(Ai, np.pi/2)
        if dt > (angle_thr): continue

        pts = []
        pts.append(p0)
        pts.append(p1)
        pts.append(p2)
        pts.append(p3)
        # check overlap
        normal = np.array([0, 1])
        poly = utils.sort_rectangle(np.array(pts))
        aspect = utils.aspect_ratio(pts)
        if aspect > 1.0: continue

        # Project the segment centers along the normal defined by the mean angle.
        projected_ends = np.array([np.dot(endp, normal) for endp in poly])
        d1 = np.maximum(projected_ends[0], projected_ends[1]) - np.minimum(projected_ends[0], projected_ends[1])
        d2 = np.maximum(projected_ends[2], projected_ends[3]) - np.minimum(projected_ends[2], projected_ends[3])
        order = np.argsort(projected_ends)
        em = order[0]
        en = order[3]
        pd = np.maximum(projected_ends[em], projected_ends[en]) - np.minimum(projected_ends[em], projected_ends[en])
        overlap = pd / (d1 + d2)
        if overlap > 1.0: continue
        mLi = get_mid_point(Li)
        mLj = get_mid_point(Lj)
        dist = distance.euclidean(mLi,mLj)
        check = dist < dis_thr and dis_thr > 30
        if not check: continue
        qa = utils.area(np.array(poly))
        print(('qa-area:', qa, ' overlaps ', overlap, ' length ', dist, ' Angle ', math.degrees(Ai) ))

        indexes.append((i,j))
        mids.append([mLi,mLj])
        quads.append(np.array(poly))

    return indexes, mids, quads


