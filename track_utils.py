import cv2
import numpy as np
import operator
import sys
def initColor(cap):
    print ("select object to track")
    lower=np.zeros([3])
    upper=np.zeros([3])
    upper[1]=255
    upper[2]=255
    (rows,cols)=cap.getFrameSize()
    mask_3d=np.zeros((rows,cols,3),dtype=np.uint8)
    masked_roi=np.zeros((rows,cols,3),dtype=np.uint8)
    esc_count=0
    roi=np.zeros([2,2])
    while True:
        frame=cap.getFrame()
        if (cap.getColor()<0).all():
            cap.showFrame()
            key=cv2.waitKey(1)
            if key == ord("q") or key==27:
        
                if ( esc_count ==0):
                    print ("object not initialized. Press esc or 'q' again to exit")
                    esc_count+=1
                elif( esc_count ==1):
                    sys.exit(0);
            continue
        color=cap.getColor() 
        color_hsv=cv2.cvtColor(color.reshape([1,1,3]),cv2.COLOR_BGR2HSV)
        color_hsv=color_hsv[0][0]
        lower=color_hsv
        lower[0]=color_hsv[0]-10
        
        upper[0]=color_hsv[0]+20
        if lower[0]<1:
            lower[0]=1
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv=cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
        lower=lower.astype(np.uint8)
        upper=upper.astype(np.uint8)
        mask=cv2.inRange(hsv,lower,upper)
        mask_3d=cv2.merge((mask,mask,mask))
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle
            c = max(cnts, key=cv2.contourArea)
            bounding_rect = cv2.boundingRect(c)
            area = bounding_rect[2] * bounding_rect[3]

            if area > 100:
                masked_roi.fill(0)
                masked_roi[mask_3d == 255] = frame[mask_3d == 255]

                [x1, y1, w1, h1] = bounding_rect
                roi = frame[y1:y1 + w1, x1:x1 + h1]
        cap.showFrame(masked_roi)
        
        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key or 'esc' is pressed, stop the loop
        #if key == ord("q") or key==27:
        if(roi.astype(np.float).sum()==0):
            print ("object not initialized. Try again")
            cap._setColor(np.asarray([-1,-1,-1]))
            continue
        else:
            break
    return bounding_rect,roi

def findLargestContour(img):
    if len(img.shape)==3:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cnts=cv2.findContours(img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(cnts)>0:
        c=max(cnts,cv2.contourArea)
    return c
        
