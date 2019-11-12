import os
import numpy as np
import cv2
import sys
from pathlib import Path
import pickle

from matplotlib import pyplot as plt
from color_histogram_classifier import HistogramColorClassifier


class Scope(object):
    def __init__(self, ax, maxt, dt):
        self.ax = ax
        self.dt = dt
        self.maxt = maxt
        self.tdata = [0]
        self.ydata = [0]
        self.line = Line2D(self.tdata, self.ydata)
        self.ax.add_line(self.line)
        self.ax.set_ylim(0.0, 1.0)
        self.ax.set_xlim(0, self.maxt)

    def update(self, y):
        lastt = self.tdata[-1]
        if lastt > self.tdata[0] + self.maxt:  # reset the arrays
            self.tdata = [self.tdata[-1]]
            self.ydata = [self.ydata[-1]]
            self.ax.set_xlim(self.tdata[0], self.tdata[0] + self.maxt)
            self.ax.figure.canvas.draw()

        t = self.tdata[-1] + self.dt
        self.tdata.append(t)
        self.ydata.append(y)
        self.line.set_data(self.tdata, self.ydata)
        return self.line,self.ax


    def t_data(self):
        self.t_data

    def y_data(self):
        self.y_data




#Defining the classifier
my_classifier = HistogramColorClassifier(channels=[0, 1, 2], hist_size=[128, 128, 128],
                                         hist_range=[0, 256, 0, 256, 0, 256], hist_type='BGR')


color_histogram_feature_file = "./projects/pad.pickle"
hist_1 = None
if (not os.path.exists(color_histogram_feature_file)):
    model_1 = cv2.imread('./projects/wiic/images/pad.png')  # Pad
    hist_1 = my_classifier.generateModelHistogram(model_1)
    # serialize the VP-Tree to disk
    print("[INFO] serializing ...")
    f = open(color_histogram_feature_file, "wb")
    f.write(pickle.dumps(hist_1))
    f.close()
else:
    print("[INFO] loading model histogram...")
    hist_1 = pickle.loads(open(color_histogram_feature_file, "rb").read())

if hist_1 is None:
    print("[ERROR] failed to create or rule loading model histogram...")
else:
    my_classifier.addModelByHistogram(hist_1)
    print("[INFO] Model added ...")




if __name__ == "__main__":
    if len(sys.argv) < 2:
        exit(1)
    print(sys.argv[1])
    if Path(sys.argv[1]).is_file():
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
                # only needed if running on Windows
                # frame = cv2.flip(frame,0)
                comparison_array = my_classifier.returnHistogramComparisonArray(frame, method="intersection")
                y = np.log(comparison_array[0])
                y = 1.0 / np.fabs(y)
                intersection.append(y)
                print((fn,y))
                fn = fn+1

                # write the flipped frame
                #out.write(frame)

                cv2.imshow('frame',frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
                    break
            else:
                break

        x = np.arange(len(intersection))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, intersection, color='tab:blue')
        ax.set_xlim([0, fc])
        ax.set_ylim([0, 1])

        plt.show()
        # Release everything if job is finished
        # if we were writing out.release()
        cap.release()
        cv2.destroyAllWindows()
