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
import skimage.filters as sk_filters
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
    settings['vertical_horizon_norm'] = 0.5
    settings['write_frames_path'] = None
    # Other Choices 'hue' or 'gray'
    settings['display_source'] = 'native_color'
    settings['expected_minimum_size'] = [18,30]
    settings['display_frame_delay_seconds'] = -1 # -1 means dont delay
    settings['display_click_after_frame'] = False
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


class Time:
  """
  Class for displaying elapsed time.
  """

  def __init__(self):
    self.start = datetime.now()

  def elapsed_display(self):
    time_elapsed = self.elapsed()
    print("Time elapsed: " + str(time_elapsed))

  def elapsed(self):
    self.end = datetime.now()
    time_elapsed = self.end - self.start
    return time_elapsed
  
  

def processing_info(np_arr, name=None, elapsed=None):
  """
  Display information (shape, type, max, min, etc) about a NumPy array.
  Args:
    np_arr: The NumPy array.
    name: The (optional) name of the array.
    elapsed: The (optional) time elapsed to perform a filtering operation.
  """

  if name is None:
    name = "NumPy Array"
  if elapsed is None:
    elapsed = "---"

  print("%-20s | Time: %-14s  Type: %-7s Shape: %s" % (name, str(elapsed), np_arr.dtype, np_arr.shape))

def _draw_str(dst, target, s, scale=1.0):
    x, y = target
    cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_COMPLEX_SMALL, scale, (255,255,255), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, scale, (128, 128, 128), lineType=cv2.LINE_AA)


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


def dot(v,w):
    x,y,z = v
    X,Y,Z = w
    return x*X + y*Y + z*Z

def length(v):
    x,y,z = v
    return math.sqrt(x*x + y*y + z*z)

def vector(b,e):
    x,y,z = b
    X,Y,Z = e
    return (X-x, Y-y, Z-z)

def unit(v):
    x,y,z = v
    mag = length(v)
    return (x/mag, y/mag, z/mag)

def distance(p0,p1):
    return length(vector(p0,p1))

def scale(v,sc):
    x,y,z = v
    return (x * sc, y * sc, z * sc)

def add(v,w):
    x,y,z = v
    X,Y,Z = w
    return (x+X, y+Y, z+Z)


# 1  Convert the line segment to a vector ('line_vec').
# 2  Create a vector connecting start to pnt ('pnt_vec').
# 3  Find the length of the line vector ('line_len').
# 4  Convert line_vec to a unit vector ('line_unitvec').
# 5  Scale pnt_vec by line_len ('pnt_vec_scaled').
# 6  Get the dot product of line_unitvec and pnt_vec_scaled ('t').
# 7  Ensure t is in the range 0 to 1.
# 8  Use t to get the nearest location on the line to the end
#    of vector pnt_vec_scaled ('nearest').
# 9  Calculate the distance from nearest to pnt_vec_scaled.
# 10 Translate nearest back to the start/end line.


def euclidean(x, y):
    """Returns the Euclidean distance between the vectors ``x`` and ``y``.

    Each of ``x`` and ``y`` can be any iterable of numbers. The
    iterables must be of the same length.

    """
    return sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))

def projection_point_on_a_line(point, line):
    (x1, y1, x2, y2) = line
    x, y = point

    # 1. Find parametric equation of general point on the line
    x_p = [x1, (x2 - x1)]
    y_p = [y1, (y2 - y1)]

    # 2. Find vector of point to line
    A = [
        [x - x_p[0], -x_p[1]],
        [y - y_p[0], -y_p[1]]
    ]
    # 3. Find vector parallel to line
    B = [
        x2 - x1,
        y2 - y1
    ]

    # 4. Find t
    t = (A[0][0] * B[0] + A[1][0] * B[1]) / (-A[0][1] * B[0] - A[1][1] * B[1])

    # 5. Find point on line
    final_point = (x_p[0] + x_p[1] * t, y_p[0] + y_p[1] * t)

    d = euclidean(point, final_point)
    return d, final_point


def point_GT (p,q):
    if p[0] > q[0]: return True
    elif p[0] == q[0] and p[1] > q[1]: return True
    return False
def point_EQ (p,q):
    return p[0] == q[0] and p[1] == q[1]

def point_GE (p,q):
    return point_GT(p,q) or point_EQ(p,q)


   
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
def compute_lines(image, mask_region, center, expected_orientation, length_limit, vertical_horizon):
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

    return filter_lines(lines, mask_region, center, expected_orientation, length_limit, vertical_horizon)


'''
Line Filtering 
mask region is a rectangle in tl image system
represented by tl and br coords

'''
def filter_lines(lines, mask_region, center, expected_orientation, expected_size, vertical_horizon):
    N = len(lines)
    p1 = np.column_stack((lines[:, :2],
                          np.ones(N, dtype = np.float32)))
    p2 = np.column_stack((lines[:, 2:],
                          np.ones(N, dtype = np.float32)))
    dx = p1[:, 0] - p2[:, 0]
    dy = p1[:, 1] - p2[:, 1]
    xc = (p1[:, 0] + p2[:, 0]) / 2.0
    yc = (p1[:, 1] + p2[:, 1]) / 2.0
    
    # setup region for line processing
    high = vertical_horizon
    high = high / 2
    low = vertical_horizon + expected_size[1]
    mask = (yc > high) & (yc < low)
 
#    mask = yc > vertical_horizon
    lines = lines[mask]
    dx = dx[mask]
    dy = dy[mask]
    xc = xc[mask]
    yc = yc[mask]
    directions = (np.arctan2(dy, dx) + np.pi)
    lengths = np.sqrt(dx * dx + dy * dy)
    mask = np.mod(directions, expected_orientation) < 0.2
    lines = lines[mask]
    dx = dx[mask]
    dy = dy[mask]
    xc = xc[mask]
    yc = yc[mask]
    directions = directions[mask]
    
    N = len(lines)
    out = np.zeros((N,N), dtype='float')
    iterator = itertools.combinations(range(N), 2)
    rects = []
    for i, j in iterator:
        dd = cosine_scipy(lines[i], lines[j])
        if dd > 0.0091: continue
        # findout which one is wider
        wide, short = i, j
        if lengths[wide] < lengths[short]:
            wide, short = j, i
        aspect = lengths[wide] / lengths[short]
        if aspect > 3.0 : continue
        
        (x1,y1,x2,y2) = lines[wide]
        p3 = np.array((x1,y1))
        p4 = np.array((x2,y2))
        d1,p1 = projection_point_on_a_line(p3,lines[short])
        d2,p2 = projection_point_on_a_line(p4,lines[short])
        if math.fabs(d1-d2) > 3: continue
        aspect = lengths[wide] / d1
        if aspect > 3.0 or aspect < 0.333 : continue
        
        points = [p1,p2,p3,p4]
        points = np.asarray(points, dtype=np.float32)
        rr = cv2.minAreaRect(points)
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


def filter_hysteresis_threshold(np_img, low=50, high=100, output_type="uint8"):
  """
  Apply two-level (hysteresis) threshold to an image as a NumPy array, returning a binary image.
  Args:
    np_img: Image as a NumPy array.
    low: Low threshold.
    high: High threshold.
    output_type: Type of array to return (bool, float, or uint8).
  Returns:
    NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above hysteresis threshold.
  """
  
  hyst = sk_filters.apply_hysteresis_threshold(np_img, low, high)
  if output_type == "bool":
    pass
  elif output_type == "float":
    hyst = hyst.astype(float)
  else:
    hyst = (255 * hyst).astype("uint8")

  return hyst


def filter_threshold(np_img, threshold, output_type="uint8"):
  """
  Return mask where a pixel has a value if it exceeds the threshold value.
  Args:
    np_img: Binary image as a NumPy array.
    threshold: The threshold value to exceed.
    output_type: Type of array to return (bool, float, or uint8).
  Returns:
    NumPy array representing a mask where a pixel has a value (T, 1.0, or 255) if the corresponding input array
    pixel exceeds the threshold value.
  """
  t = Time()
  result = (np_img > threshold)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  processing_info(result, "Threshold", t.elapsed())
  return result

def filter_grays(rgb, tolerance=15, output_type="uint8"):
  """
  Create a mask to filter out pixels where the red, green, and blue channel values are similar.
  Args:
    np_img: RGB image as a NumPy array.
    tolerance: Tolerance value to determine how similar the values must be in order to be filtered out
    output_type: Type of array to return (bool, float, or uint8).
  Returns:
    NumPy array representing a mask where pixels with similar red, green, and blue values have been masked out.
  """
  t = Time()
  (h, w, c) = rgb.shape

  rgb = rgb.astype(np.int)
  rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
  rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
  gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
  result = ~(rg_diff & rb_diff & gb_diff)

  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  processing_info(result, "Filter Grays", t.elapsed())
  return result



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


