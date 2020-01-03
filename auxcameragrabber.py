import os
import numpy as np
import cv2
import sys
import math
from pathlib import Path
import pickle
from padChecker import padChecker
from find_fiducial_wii import fiducialFinder

import time
import argparse  # provide interface for calling this script
import utils
from find_lines_lsd import lsd_lines
from wiitricity_settings import initialize_settings


def process_frame(frame, checker, settings_dict):
    settings = settings_dict
    settings['frame_count'] = settings['frame_count'] + 1
    tl = settings['frameTopLeft']
    br = settings['frameBottomRight']
    frame = cv2.medianBlur(frame, 7)
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
        if utils.circleContainsBoundingRect(settings['active_center'], radius_2, bounding_rect):
            bbs.append(bounding_rect)
            contours.append(cnt)
            cv2.drawContours(display, cnt, -1, (128, 128, 128), -1)

    def draw_cross(img, center, color, d):
        cv2.line(img,
                 (center[0] - d, center[1]), (center[0] + d, center[1]), color, 1, cv2.LINE_AA, 0)
        cv2.line(img,
                 (center[0] , center[1] - d), (center[0] , center[1] + d),color, 1, cv2.LINE_AA, 0)

    for bb in bbs:
        area = bb[2] * bb[3]
        min_radius = math.sqrt(area / math.pi)
        cx = int(bb[0] + bb[2] / 2)
        cy = int(bb[1] + bb[3] / 2)
        draw_cross(display, (cx, cy), (0, 0, 255), 10)
        cv2.circle(display, (cx, cy), int(min_radius + 0.5), (255, 0, 0), -1)

    check_time = time.time()
    raw_lines = lsd_lines(gray)
    lines = []
    for line in raw_lines:
        if utils.circleContainsLine(settings['active_center'], radius_2, line):
            lines.append(line)

    lsd_time = time.time()
    lsd_time = lsd_time - check_time
    check_time = check_time - start_time
    print((check_time, lsd_time))

    for line in lines:
        angle = utils.get_line_angle(line)
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

    utils.drawString(frame, str(fcount))
    cv2.circle(display, settings['active_center'], settings['active_radius'], (0, 255, 0), 3)
    res = np.vstack((display, seethrough))
    return res


if __name__ == "__main__":

    # Get input parameters
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--source", required=False, default='camera', help="camera -- default is camera")
    ap.add_argument("-v", "--video", required=False, help="path to input video file")
    ap.add_argument("-i", "--image", required=False, help="path to input image file")
    ap.add_argument("-c", "--cache", required=False, default='./projects/wiic',
                    help="path to cache directory relative to source code directory")
    ap.add_argument("-f", "--fiducial", required=False, default='no')

    args = vars(ap.parse_args())
    for (k, v) in args.items():
        print(k, v)

    file_folder = os.path.dirname(os.path.realpath(__file__))
    is_video_file = (not (args['video'] is None)) and args['source'] == 'file' and Path(args['video']).is_file()
    is_image_file = (not (args['image'] is None)) and args['source'] == 'file' and Path(args['image']).is_file()
    is_camera = not (is_video_file or is_image_file)
    cachePath = os.path.abspath(os.path.join(file_folder, args['cache']))

    if is_video_file:
        file_name = args['video']
    elif is_image_file:
        file_name = args['image']

    if args['fiducial'] and args['fiducial'] is 'yes':
        checker = fiducialFinder(cachePath)
    else:
        checker = padChecker(cachePath)

    cap = None
    if is_video_file:
        cap = cv2.VideoCapture(file_name)  # Capture video from file
    elif is_image_file:
        image = cv2.imread(file_name)
    elif is_camera:
        cap = cv2.VideoCapture(0)  # Capture video from camera

    if is_image_file:
        # setup settings based on capture size and padding for black regions around
        shape = image.shape
        settings = initialize_settings((10, 60), (shape[1], shape[0]))
        res = process_frame(image, checker, settings)
        cv2.imshow('image', res)
        cv2.waitKey(0)

    else:
        settings = initialize_settings((10, 60),
                                       (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        ret, frame = cap.read()
        if ret == False: exit(1)
        fcount = 1
        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret: continue
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
