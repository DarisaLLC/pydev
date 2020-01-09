import sys
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import math


def get_line_angle(line):
    """Calculates the angle of a line.

    Args:
        line: Vector of x1, y1, x2, y2, i.e. two points on the line.

    Returns:
        Float, angle in degrees.
    """
    x1, y1, x2, y2 = np.array(line, dtype=np.float64)
    radians = np.arctan2(y2 - y1, x2 - x1)
    return math.degrees(radians) % 360

def point_to_point_dist(point_a, point_b):
    """Finds euclidean distance between two points.

    Args:
        point_a: Tuple (x, y) point.
        point_b: Tuple (x, y) point.

    Returns:
        Float, distance.
    """
    x1, y1 = np.array(point_a, dtype=np.float64)
    x2, y2 = np.array(point_b, dtype=np.float64)
    return math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    if not Path(sys.argv[1]).is_file() or not Path(sys.argv[1]).exists():
        print(sys.argv[1] + '  Does not exist ')
        sys.exit(1)

    video_file = sys.argv[1]
    filename = Path(video_file).name
    cap = cv.VideoCapture(video_file)
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.001,
                           minDistance = 10,
                           blockSize = 25 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (13,13),
                      maxLevel = 3,
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_frame = old_frame[53:350, 6:708]
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)


    def bgrFromHue(degrees):
        hsv = np.zeros((1, 1, 3), np.uint8)
        hsv[0, 0, 0] = ((degrees % 360) * 256) / 360.0
        hsv[0, 0, 1] = ((degrees % 90) * 256) / 90.0
        hsv[0, 0, 2] = ((degrees % 45) * 256) / 45.0
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
        tp = tuple([int(x) for x in bgr[0,0,:]])
        return tp

    fc = 1
    directions = []
    lhist = np.zeros((1,360), np.int)
    while(1):
        ret,frame = cap.read()
        if frame is None: break
        frame = frame[53:350, 6: 708]

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        if len(p1) < 1:
            continue
        if fc == 1:
            cv.imwrite('/Users/arman/tmp/first.png', frame_gray)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        # draw the tracks
        print('------>' + str(fc))
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            line = [a,b,c,d]
            angle = int(get_line_angle(line))
            dis = point_to_point_dist((a,b),(c,d)) // 1
            lhist[0][angle] = lhist[0][angle] + dis
            if dis > 0:
                print (str(angle) + '......' + str(dis))
                directions.append(angle)
            cl = bgrFromHue(angle)
            mask = cv.line(mask, (a,b),(c,d), cl, 2)
            frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
        print('<------' + str(fc))
        img = cv.add(frame,mask)
        cv.namedWindow('frame', cv.WINDOW_NORMAL)
        cv.imshow('frame',img)
        fc = fc + 1
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

    _ = plt.hist(lhist , fc='k', ec='k')  # arguments are passed to np.histogram
    plt.title(filename)
    plt.show()
    cv.waitKey(0)
    cv.destroyAllWindows()
    cap.release()