import numpy as np
import sys
import math
import cv2 as cv
import datetime
import logging
from datetime import datetime
from enum import Enum, unique

@unique
class State(Enum):
    eRight=-1,
    eLeft=+1,
    eSame=0,
    eUnknown=-2

'''
Initialization of common settings for WiiTricity image & videos

Image data location within a stored image or frame of video
frame_tl is the topleft corner of image data in the stored image or video frame
capture_size is the image size 

'''
video_rois = {'one' : dict(row_low=53,row_high=350,column_low=6,column_high=708),
              'hd2' : dict(row_low=0,row_high=759,column_low=0,column_high=1919),
              'hd' : dict(row_low=0,row_high=519,column_low=0,column_high=1279)}

def initialize_settings_from_video_roi(video_roi):
    frame_tl = (video_roi['column_low'], video_roi['row_low'])
    capture_size = (video_roi['column_high']-video_roi['column_low'],video_roi['row_high']-video_roi['row_low'])
    return initialize_settings(frame_tl, capture_size)

def initialize_settings(frame_tl, capture_size):
    settings = {'frameTopLeft': frame_tl, 'active_center_norm': (0.5, 0.5), 'active_radius_norm': 0.4,
                'capture_size': capture_size, 'cache':'.'}

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
                     maxLevel=2,
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
        (0, top), (cols-1, top), (cols - 1, rows - 1), (0, rows - 1)]], np.int32)

    cv.fillConvexPoly(mask, poly, 255)
    return mask


def _draw_str(dst, target, s, scale=1.0):
    x, y = target
    cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, scale, (0, 255, 0), thickness = 2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, scale, (128,128,128), lineType=cv.LINE_AA)


# create logger with 'wiirunner'
def get_logger():
    logfilename = 'wiirunner'+datetime.now().strftime("-%d-%m-%Y_%I-%M-%S_%p")+'.log'
    logger = logging.getLogger('wiirunner')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(logfilename)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
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
