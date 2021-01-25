import numpy as np
import sys
import math
import cv2

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
                'capture_size': capture_size}
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
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # params for ShiTomasi corner detection
    settings['feature_params'] = dict(maxCorners=8000,
                          qualityLevel=0.01,
                          minDistance=9,
                          blockSize=7,
                          useHarrisDetector=False,
                          k=0.04)
    settings['max_distance'] = 25
    settings['min_features'] = 300



    return settings


