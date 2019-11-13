import os
import numpy as np
import cv2
import sys
from pathlib import Path
import pickle
from padChecker import padChecker
from matplotlib import pyplot as plt



if __name__ == "__main__":

    if len(sys.argv) < 3:
        exit(1)
    print(sys.argv[1])
    print(sys.argv[2])
    if Path(sys.argv[1]).is_file():
        checker = padChecker(cachePath=sys.argv[2])
        file_name = sys.argv[1]
        cap = cv2.VideoCapture(file_name) # Capture video from camera

        # Get the width and height of frame
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define the codec and create VideoWriter object
        #fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the lower case
        #out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                y = checker.check(frame)
                maxi = np.max(checker.history())
                x = np.arange(len(checker.history()))
                vals = checker.history()
                vals = np.multiply(height,vals)
                vals = np.subtract(height,vals)
                pts = np.vstack((x,vals)).astype(np.int32).T
                cv2.polylines(frame, [pts],isClosed=False, color=(255,0,0))

                cv2.imshow('frame',frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
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

