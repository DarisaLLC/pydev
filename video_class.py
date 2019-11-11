import cv2
import numpy as np

class CamVideo:
    
    def __init__(self):
        self._win_name="video"
        nocolor=np.asarray([-1,-1,-1])
        self.color=nocolor
        downsample_factor=1 
        self.cap=cv2.VideoCapture(0)
        self.height=self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.width=self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,int(self.height/downsample_factor))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,int(self.width/downsample_factor))
        cv2.namedWindow(self._win_name)
        cv2.moveWindow(self._win_name,100,100)
        cv2.setMouseCallback(self._win_name,self._onMouse,None)
    
    def _setFrameSize(self,height,width):
        self.height=height
        self.width=width
    
    def getFrameSize(self):
        return((self.height,self.width))

    def _onMouse(self,event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDBLCLK:
            frame_size=self.getFrameSize()
            #print y,x
            self.setLastClickXY(x,y)
            self._setColor(self.frame[y,x])
   
    def setLastClickXY(self,x,y):
        self.x=x
        self.y=y
    
    def getLastClickXY(self):
        return np.asarray([self.x,self.y])

    def _setColor(self,color):
       self.color=color
  
    def getColor(self):
        return np.asarray(self.color)
        
    def getFrame(self):
        (grabbed,self.frame)=self.cap.read()
        if grabbed:
            self.frame=cv2.flip(self.frame,1)
            return self.frame
        print ("no frame grabbed")
        return None
    

    def showFrame(self,frame=None):
        if frame==None:
            frame=self.frame
        cv2.imshow(self._win_name,frame)
