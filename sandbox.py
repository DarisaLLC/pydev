import random
import csv
import numpy as np
import matplotlib as pl
from matplotlib import pyplot as plt
import stats

def lazy_property(fn):
    '''Decorator that makes a property lazy-evaluated.
    '''
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


class Person:
    def __init__(self, name, occupation):
        self.name = name
        self.occupation = occupation

    @lazy_property
    def relatives(self):
        # Get all relatives
        relatives = 'relatives'
        return relatives


from tkinter import *
from tkinter import filedialog
import tkinter
import os


class Chooser:
    def __init__(self, master):
        self.resolution = StringVar()
        self.resolution.set("360p")
        self.filename = StringVar()

        label = Label(master, text="Select Target Resolution")
        label.pack(fill=X, padx=10, pady=10)

        option_res = OptionMenu(master, self.resolution, "240p", "360p", "480p", "720p", "1080p", command=self.selected)
        option_res.pack(padx=10, pady=10)

        button_choose = Button(master, text='Choose File', command=self.choose_file)
        button_choose.pack(fill=X, padx=10, pady=40)

        button_start = Button(master, text='Start', command=self.start)
        button_start.pack(fill=X, padx=5)

    def selected(self, res):
        print("Resolution " + self.resolution.get() + " is chosen.")

    def choose_file(self):
        self.filename = filedialog.askopenfilename(initialdir="/home",
                                                   filetypes=[('All', '*'), ('mp4', '*.mp4'), ('mpeg', '*mpeg')])
        print("File " + self.filename + " is chosen.")

    def start(self):
        print("Resizer Converting your video...")
        # print "../core/cmake-build-debug/VideoResAdjuster " + self.filename + " " + self.resolution.get()
        os.system("../core/cmake-build-debug/VideoResAdjuster " + self.filename + " " + self.resolution.get())
        print("Finished.")

#
# master = Tk()
# master.minsize(width=320, height=240)
# master.title("Video Resolution Adjuster")
# chooser = Chooser(master)
#
# mainloop()

from sympy import *
from itertools import combinations



import numpy as np
import cv2

from multiprocessing import Process, Queue
import time


# from common import clock, draw_str, StatValue
# import video

class Canny_Process(Process):

    def __init__(self, frame_queue, output_queue):
        Process.__init__(self)
        self.frame_queue = frame_queue
        self.output_queue = output_queue
        self.stop = False
        # Initialize your face detectors here

    def get_frame(self):
        if not self.frame_queue.empty():
            return True, self.frame_queue.get()
        else:
            return False, None

    def stopProcess(self):
        self.stop = True

    def canny_frame(self, frame):
        # some intensive computation...
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 100)

        # To simulate CPU Time
        #############################
        for i in range(1000000):
            x = 546 * 546
            res = x / (i + 1)
        #############################
        'REPLACE WITH FACE DETECT CODE HERE'

        if self.output_queue.full():
            self.output_queue.get_nowait()
        self.output_queue.put(edges)

    def run(self):
        while not self.stop:
            ret, frame = self.get_frame()
            if ret:
                self.canny_frame(frame)


if __name__ == '__main__':
    
    from skimage.util import img_as_float
    from skimage.util import img_as_ubyte
    from skimage.feature import peak_local_max
    
    image = np.zeros((88,66), dtype="uint8")
    image[22:66,0:42] = 255

    f, axs = plt.subplots(2, 2, figsize=(20, 10), frameon=False,
                          subplot_kw={'xticks': [], 'yticks': []})
    axs[0, 0].imshow(image, cmap=plt.cm.gray)
    plt.show()
    three = cv2.imread('/Users/arman/Pictures/nine.png')
    nine, blah, foo = cv2.split(three)
    uuu = cv2.imread('/Users/arman/Pictures/uuu.png')
    u, lah, oo = cv2.split(uuu)

    res = cv2.matchTemplate(img_as_ubyte(nine), img_as_ubyte(u), cv2.TM_CCORR_NORMED)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    local_maxi = peak_local_max(img_as_float(res), indices = True, min_distance = 20, exclude_border = False)
    
    for loc in local_maxi:
        cv2.circle(three, (loc[0],loc[1]), 5, (0,0,255), 2)
        
    
    cv2.imshow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', three)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # frame_sum = 0
    # init_time = time.time()
    #
    #
    # def put_frame(frame):
    #     if Input_Queue.full():
    #         Input_Queue.get_nowait()
    #     Input_Queue.put(frame)
    #
    #
    # def cap_read(cv2_cap):
    #     ret, frame = cv2_cap.read()
    #     if ret:
    #         put_frame(frame)
    #
    #
    # cap = cv2.VideoCapture(0)
    #
    # threadn = cv2.getNumberOfCPUs()
    #
    # threaded_mode = True
    #
    # process_list = []
    # Input_Queue = Queue(maxsize=5)
    # Output_Queue = Queue(maxsize=5)
    #
    # for x in range((threadn - 1)):
    #     canny_process = Canny_Process(frame_queue=Input_Queue, output_queue=Output_Queue)
    #     canny_process.daemon = True
    #     canny_process.start()
    #     process_list.append(canny_process)
    #
    # ch = cv2.waitKey(1)
    # cv2.namedWindow('Threaded Video', cv2.WINDOW_NORMAL)
    # while True:
    #     cap_read(cap)
    #
    #     if not Output_Queue.empty():
    #         result = Output_Queue.get()
    #         cv2.imshow('Threaded Video', result)
    #         ch = cv2.waitKey(5)
    #
    #     if ch == ord(' '):
    #         threaded_mode = not threaded_mode
    #     if ch == 27:
    #         break
    # cv2.destroyAllWindows()
    #
    #

