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


from coloralgo import CenterOfIntensity
from opencv_utils import drawString


from vp import geom_tools


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
    min_line_length = min_line_length * diagonal
    max_line_length = max_line_length * diagonal

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
    if Path(sys.argv[1]).is_file():
        checker = padChecker(cachePath=sys.argv[2])
        file_name = sys.argv[1]
        cap = cv2.VideoCapture(0) # Capture video from camera

        pad = (10,60)
        # Get the width and height of frame
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5) - pad[0]*2
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5) - pad[1]*2
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


        ret, frame = cap.read()
        if ret == False: exit(1)
        # Define the codec and create VideoWriter object
        #fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the lower case
        #out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))
        lastgray = None
        fcount = 1
        while(cap.isOpened()):
            ret, frame = cap.read()
            fcount = fcount + 1
            _maxLoc = (0,0)
            if ret == True:
                tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = tmp[pad[1]:pad[1]+height, pad[0]:pad[0]+width]
                com = CenterOfIntensity(gray)
                print((com))

                y, mask = checker.check(frame[pad[1]:pad[1]+height, pad[0]:pad[0]+width])

                gray = cv2.pyrUp(gray)
                gray = cv2.pyrUp(gray)
                gray = cv2.pyrDown(gray)
                gray = cv2.pyrDown(gray)

                if not (lastgray is None):
                    method = cv2.TM_CCORR_NORMED
                    roi = lastgray[10:height-20, 10:width-20]
                    res = cv2.matchTemplate(gray, roi, cv2.TM_CCOEFF_NORMED)
                    _minVal, _maxVal, _minLoc, _maxLoc = cv2.minMaxLoc(res, None)
                    rr = _maxVal*_maxVal
                    print((_maxVal*_maxVal,_maxLoc))
#                    if rr > 0.99:
                    lastgray:gray
                else:
                    lastgray = gray

                lines = lsd_lines(gray)

                display = frame[pad[1]: pad[1] + height, pad[0]: pad[0] + width]
                cv2.line(display, (0,0), _maxLoc, (0, 0, 255), 2)

                for line in lines:
                    angle = geom_tools.get_line_angle(line)
                    vert = angle%90
                    horz = angle%180
                    x1, y1, x2, y2 = line
                    cv2.line(display, (x1, y1), (x2, y2), (0, vert*255/90., horz*255/180.), 2)

                maxi = np.max(checker.history())
                x = np.arange(len(checker.history()))
                vals = checker.history()
                vals = np.multiply(height, vals)
                vals = np.subtract(height, vals)
                pts = np.vstack((x, vals)).astype(np.int32).T
                cv2.polylines(display, [pts], isClosed=False, color=(255, 0, 0))

                # vp_to_lines, outlier_lines = vp_finder.find_vanishing_points_in_image (frame)
                # for vp in vp_to_lines:
                #     print('Vanishing point found at: (%s, %s)' % (vp[0], vp[1]))

                drawString(frame, str(fcount))

                cv2.imshow('frame',display)
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

