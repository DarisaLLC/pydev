

import cv2 as cv
import numpy as np
import sys
from pathlib import Path

if __name__ == "__main__":

    if len(sys.argv) < 2:
        exit(1)
    print(sys.argv[1])
    if Path(sys.argv[1]).is_file():
        file_name = sys.argv[1]
        cap = cv.VideoCapture(file_name) # Capture video from camera
        ret, frame1 = cap.read()
        prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        
        while True:
            ret, frame2 = cap.read()
            if not ret: break
            next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

            # prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
            flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.75, 3, 5, 7, 5, 1.1, 0)
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            vert = ang % (np.pi / 2.0)
            vert = vert < (np.pi / 16.)
            hsv[..., 0] = ang * 180 / np.pi
            nmag = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            nmag[vert] = 0.0
            hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

            cv.imshow('frame2', mag)
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
            elif k == ord('s'):
                cv.imwrite('opticalfb.png', frame2)
                cv.imwrite('opticalhsv.png', bgr)
            prvs = next
        
        cap.release()
        cv.destroyAllWindows()
