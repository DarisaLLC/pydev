import numpy as np
import cv2 as cv
import argparse
from videoio import getMetaData
import json

parser = argparse.ArgumentParser(description='meanshift-tracking')
parser.add_argument('image', type=str, help='path to image file')
parser.add_argument('fiducial', type=str, help='path to fiducial file')
args = parser.parse_args()

mdata = getMetaData(args.image)
print(mdata.keys())
print(json.dumps(mdata["video"], indent=4))

fid = cv2.imread(args.fiducial)
fidhsv = cv2.cvtColor(fid,cv2.COLOR_BGR2HSV)
fid_hist = cv.calcHist([hsv_roi],[0],None,[180],[0,180])
cv.normalize(fid_hist,fid_hist,0,255,cv.NORM_MINMAX)

cap = cv.VideoCapture(args.image)

# take first frame of the video
ret,frame = cap.read()


# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

i = 0
while(1):
    ret, frame = cap.read()

    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],fid_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x,y,w,h = track_window
        img2 = cv.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv.imshow('img2',img2)
        i = i + 1

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break

while(1):
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break