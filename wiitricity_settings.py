import numpy as np
import sys
import math
import utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

'''
Initialization of common settings for WiiTricity image & videos

Image data location within a stored image or frame of video
frame_tl is the topleft corner of image data in the stored image or video frame
capture_size is the image size 

'''
video_rois = {'one' : dict(row_low=53,row_high=350,column_low=6,column_high=708)}

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
                          minDistance=5,
                          blockSize=3,
                          useHarrisDetector=False,
                          k=0.04)
    settings['max_distance'] = 15
    settings['min_features'] = 150



    return settings



def draw_lane(im, left_fitx, right_fitx, leftw, rightw):
    """ draw lane and sliding windows on binary warped image """
    out_img = np.dstack((im, im, im)) * 255

    # draw the lines
    ploty = np.linspace(0, im.shape[0] - 1, im.shape[0])
    left_pts = np.dstack((left_fitx, ploty)).astype(np.int32)
    right_pts = np.dstack((right_fitx, ploty)).astype(np.int32)

    # draw the curve through detected lane line segments with yellow color
    cv2.polylines(out_img, left_pts, False, (255, 255, 0), 4)
    cv2.polylines(out_img, right_pts, False, (255, 255, 0), 4)

    # draw sliding windows on both lanes with color Cyan
    for low, high in leftw:
        cv2.rectangle(out_img, low, high, (0, 255, 255), 3)

    for low, high in rightw:
        cv2.rectangle(out_img, low, high, (0, 255, 255), 3)

    return out_img


def draw_region(im, left_fitx, right_fitx):
    """ draw lane area on binary warped image """
    margin = 50
    ploty = np.linspace(0, im.shape[0] - 1, im.shape[0])

    left_w1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_w2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_pts = np.hstack((left_w1, left_w2))

    right_w1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_w2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_pts = np.hstack((right_w1, right_w2))

    # Create RGB image from binary warped image
    region_img = np.dstack((im, im, im)) * 255

    # Draw the lane onto the warped blank image
    cv2.fillPoly(region_img, np.int_([left_pts]), color=(255, 0, 255), lineType=4)
    cv2.fillPoly(region_img, np.int_([right_pts]), color=(255, 0, 255), lineType=4)

    return region_img


def draw_area(im, left_fitx, right_fitx, M):
    """ draw the area between detected left lane and right lane """
    color = np.zeros_like(im)
    ploty = np.linspace(0, im.shape[0] - 1, im.shape[0])

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color, np.int_([pts]), (51, 0, 102))

    # newwarp = np.zeros_like(im)
    newwarp = cv2.warpPerspective(color, M, (im.shape[1], im.shape[0]), flags=cv2.WARP_INVERSE_MAP)

    result = cv2.addWeighted(im, 1, newwarp, 0.3, 0)
    return result
