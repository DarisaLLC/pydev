import os
import numpy as np
import cv2
import sys
import math
from pathlib import Path
import pickle
from padChecker import padChecker
from matplotlib import pyplot as plt
from vp import vp_ransac
from vp import vp_finder
import time


from coloralgo import CenterOfIntensity
from opencv_utils import drawString


from vp import geom_tools


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
    min_line_length = 5 #min_line_length * diagonal
    max_line_length = 66 #max_line_length * diagonal
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
    return [l[0] for (i, l) in enumerate(lines)
            if max_line_length > line_lengths[i] > min_line_length and
            precisions[i] > min_precision]


if __name__ == "__main__":

    if len(sys.argv) < 3:
        exit(1)
    print(sys.argv[1])
    print(sys.argv[2])

    ## Geometric constants ( to be fetched from some config file )
    pad_pixels_minimum_size = 5

    if Path(sys.argv[1]).is_file():
        checker = padChecker(cachePath=sys.argv[2])
        file_name = sys.argv[1]
        cap = cv2.VideoCapture(file_name) # Capture video from camera

        pad = (10,60)
        # Get the width and height of frame
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5) - pad[0]*2
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5) - pad[1]*2
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


        ret, frame = cap.read()
        if ret == False: exit(1)
        lastgray = None
        fcount = 1
        while(cap.isOpened()):
            ret, frame = cap.read()
            fcount = fcount + 1
            _maxLoc = (0,0)
            if ret == True:
                display = frame[pad[1]: pad[1] + height, pad[0]: pad[0] + width]
                lab_image  = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                # Split LAB channels
                gray, a, b = cv2.split(lab_image)
                gray = gray[pad[1]:pad[1]+height, pad[0]:pad[0]+width]

                start_time = time.time()
                y, mask, seethrough = checker.check(frame[pad[1]:pad[1] + height, pad[0]:pad[0] + width])

                cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

                if len(cnts) > 0:
                    # find the largest contour in the mask, then use
                    # it to compute the minimum enclosing circle
                    c = max(cnts, key=cv2.contourArea)
                    bounding_rect = cv2.boundingRect(c)
                    area = bounding_rect[2] * bounding_rect[3]
                    min_radius = math.sqrt(area/math.pi)

                check_time = time.time()
                lines = lsd_lines(gray)
                lsd_time = time.time()
                lsd_time = lsd_time - check_time
                check_time = check_time - start_time
                print((check_time,lsd_time))

                for line in lines:
                    angle = geom_tools.get_line_angle(line)
                    vert = angle%90
                    horz = angle%180
                    x1, y1, x2, y2 = line
                    cv2.line(display, (x1, y1), (x2, y2), (0, vert*255/90., horz*255/180.), 2)

                vals = checker.history()
                maxi = np.max(vals)
                x = np.arange(len(vals))
                vals = np.multiply(height, vals)
                vals = np.subtract(height, vals)
                pts = np.vstack((x, vals)).astype(np.int32).T
                cv2.polylines(display, [pts], isClosed=False, color=(255, 0, 0))

                drawString(frame, str(fcount))
                cx = int(bounding_rect[0]+bounding_rect[2]/2)
                cy = int(bounding_rect[1]+bounding_rect[3]/2)
                draw_cross(display, (cx,cy),(0, 0, 255), 3)
                cv2.circle(display, (cx, cy), int(min_radius+0.5), (255, 0, 0), -1)

                res = np.vstack((display,seethrough))
                cv2.imshow('frame', res)
                kk = cv2.waitKey(1) & 0xff
                Pause = kk == ord('c')
                if kk == ord('n') or Pause == False: continue
                else:
                    if kk == ord('q'): # Hit `q` to exit
                        break
            else:
                exit(0)

        x = np.arange(len(checker.history()))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, checker.history(), color='tab:blue')
        ax.set_xlim([0, fc])
        ax.set_ylim([0, 1])

        plt.show()
        # Release everything if job is finished
        # if we were writing out.release()
        cap.release()
        cv2.destroyAllWindows()
        exit(0)

