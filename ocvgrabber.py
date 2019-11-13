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
        intersection = []
        fn = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                y = checker.check(frame)
                cv2.imshow('frame',frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
                    break
            else:
                break

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
