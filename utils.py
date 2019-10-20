import cv2
import numpy as np

def mouse_handler(event, x, y, flags, data) :

    if event == cv2.EVENT_LBUTTONDOWN :
        cv2.circle(data['im'], (x,y),3, (0,0,255), 5, 16);
        cv2.imshow("Image", data['im']);
        if len(data['points']) < 4 :
            data['points'].append([x,y])

def get_four_points(im):
    # Set up data to send to mouse handler
    data = {}
    data['im'] = im.copy()
    data['points'] = []

    #Set the callback function for any mouse event
    cv2.imshow("Image",im)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)

    # Convert array to np.array
    points = np.vstack(data['points']).astype(float)

    return points


import cv2
import numpy as np
import sys


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
