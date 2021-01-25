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
              'hd': dict(row_low = 0, row_high = 519, column_low = 0, column_high = 1279)}
              

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
    settings['reduction'] = 1
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
    settings['display_frame_delay_seconds'] = 1 # -1 means dont delay
    settings['display_click_after_frame'] = False
    settings['restrict_to_view_angle'] = True
    settings['rects_too_small_area'] = 300
    ## Synthesize, runs input video but instead of captured frames it synthesizes a moving / rotating rectangle
    settings['synthesize_test'] = False
    settings['fiducial_load'] = True
    # If set to true, that is the only operation performed
    settings['fiducial_run'] = False
    
    return settings


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

