import cv2
import numpy as np

target = cv2.imread("/Volumes/medvedev/Users/arman/tmp/fid/fid315.png")
hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)

roi = cv2.imread("/Volumes/medvedev/Users/arman/tmp/fid/fid316.png")
hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

# calculating object histogram
roihist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
hist2 = cv2.calcHist([hsvt],[0, 1], None, [180, 256], [0, 180, 0, 256] )
cv2.normalize(roihist,roihist)
cv2.normalize(hist2,hist2)
comparison = cv2.compareHist(roihist.flatten(), hist2.flatten(), cv2.HISTCMP_INTERSECT)
print(comparison)


# normalize histogram and apply backprojection
roihist2 = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
cv2.normalize(roihist2,roihist2,0,255,cv2.NORM_MINMAX)
dst = cv2.calcBackProject([hsvt],[0,1],roihist2,[0,180,0,256],1)

# Now convolute with circular disc
#disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#cv2.filter2D(dst,-1,disc,dst)

# threshold and binary AND
ret,thresh = cv2.threshold(dst,5,255,0)

thresh = cv2.merge((thresh,thresh,thresh))
res = cv2.bitwise_and(target,thresh)

res = np.vstack((target,roi,res))
cv2.imwrite('/Users/arman/tmp/res.jpg',res)
