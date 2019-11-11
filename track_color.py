#!/usr/bin/env python

import video_class
import cv2
import numpy as np
import track_utils

if __name__=="__main__":
    import sys
    messages=[ 
            "showing masked ROI",
            "showing contour",
            "bounded rectangle"]
    cap=video_class.CamVideo()
    
    _,roi=track_utils.initColor(cap)
    roi_hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    # calculating object histogram
    roi_hist = cv2.calcHist([roi_hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

    print ("init done ")
    display_count=0
    color=cap.getColor() 
    color_hsv=cv2.cvtColor(color.reshape([1,1,3]),cv2.COLOR_BGR2HSV)
    color_hsv=color_hsv[0][0]
    print ("press 'b' to toggle views")
    while True:
        frame_b=cap.getFrame()
        frame=frame_b
        #frame = cv2.GaussianBlur(frame_b, (15,15), 0)
        
        frame_hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        
        dst = cv2.calcBackProject([frame_hsv],[0,1],roi_hist,[0,180,0,255],1)
        
        
        # Now convolute with circular disc
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        cv2.filter2D(dst,-1,disc,dst)
        ret,mask = cv2.threshold(dst,150,255,cv2.THRESH_BINARY)
        mask=cv2.erode(mask,None,iterations=2)
        mask=cv2.dilate(mask,None,iterations=2)
        
        #find largest contour 
        cnts=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(cnts)>0:
            c=max(cnts,key=cv2.contourArea)
            mask=np.zeros(frame.shape,np.uint8)
            cv2.drawContours(mask,[c],0,(255,255,255),-1)
        else:
            mask = cv2.merge((mask,mask,mask))
       
        res = cv2.bitwise_and(frame_b,mask)

        if display_count==0:
            cap.showFrame(res)
        elif display_count==1:
            cv2.drawContours(frame,[c],0,(0,0,255),3)
            cap.showFrame(frame)
        else: 
            [x1,y1,w1,h1]=cv2.boundingRect(c)
            cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(0,0,255),2)
            cap.showFrame(frame)
        key = cv2.waitKey(1) & 0xFF
 
        # if the 'q' key or 'esc' is pressed, stop the loop
        if key == ord("q") or key==27:
            
            break
        if key==ord("b"):
            display_count=display_count+1
            if display_count>=3:
                display_count=0
            print (messages[display_count])
