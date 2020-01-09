import numpy as np
import cv2 as cv
import sys
from pathlib import Path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        exit(1)
    print(sys.argv[1])

    if not Path(sys.argv[1]).is_file(): exit(1)

    video_file = sys.argv[1]

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
    old_frame = cv.resize(old_frame, (0,0), None, 0.5, 0.5)
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    old_gray = old_gray [60//2: 516//2,10//2:710//2]
    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # Create a mask image for drawing purposes
    old_frame = old_frame [60//2: 516//2,10//2:710//2]
    mask = np.zeros_like(old_frame)
    while(1):
        ret,frame = cap.read()
        if frame is None:break
        frame = cv.resize(frame, (0,0), None, 0.5, 0.5)
        frame = frame[60//2: 516//2,10//2:710//2]

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv.add(frame,mask)
        cv.imshow('frame',img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
    cv.destroyAllWindows()
    cap.release()