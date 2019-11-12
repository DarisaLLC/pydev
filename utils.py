import cv2
import numpy as np



import cv2
import numpy as np
import sys

import numpy



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

def choose(x):
    return {
        's': 1,
        'r': 2,
        'b': 3,
    }[x]


def wait():
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_new(img):
    height, width, channels = img.shape
    return np.zeros((height, width, 3), np.uint8)


def meanshift_segment(img, sp, sr, option):
    src = img.copy()
    i = 0
    dpath = '/Users/arman/tmp/eye/'
    if (option == 's'):
        while (i <= sp):
            src = cv2.cvtColor(src, cv2.COLOR_BGR2LAB, src)
            dest = create_new(src)
            cv2.pyrMeanShiftFiltering(src, i, sr, dest, 2)
            dest = cv2.cvtColor(dest, cv2.COLOR_Lab2BGR, dest)
            cv2.imwrite(dpath + str(i) + '.png', dest)
            src = dest
            i = i + 2
    elif (option == 'r'):
        while (i <= sr):
            src = cv2.cvtColor(src, cv2.COLOR_BGR2LAB, src)
            dest = create_new(src)
            cv2.pyrMeanShiftFiltering(src, sp, i, dest, 2)
            dest = cv2.cvtColor(dest, cv2.COLOR_Lab2BGR, dest)
            cv2.imwrite(dpath + str(i) + '.png', dest)
            src = dest
            i = i + 2
    elif (option == 'b'):
        i = 0;
        j = 0;
        k = 0
        while (i <= sp and j <= sr):
            src = cv2.cvtColor(src, cv2.COLOR_BGR2LAB, src)
            dest = create_new(src)
            cv2.pyrMeanShiftFiltering(src, i, j, dest, 2)
            dest = cv2.cvtColor(dest, cv2.COLOR_Lab2BGR, dest)
            cv2.imwrite(dpath + str(k) + '.png', dest)
            src = dest
            i = i + 2
            j = j + 2
            k = k + 1


if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    sp = int(sys.argv[2])
    sr = int(sys.argv[3])
    option = sys.argv[4]
    meanshift_segment(img, sp, sr, option)
