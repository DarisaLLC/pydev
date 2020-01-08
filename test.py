import time
import multiprocessing
import cv2

class ImageProcessor(multiprocessing.Process):

    def __init__(self, tasks_q, results_q):
        multiprocessing.Process.__init__(self)
        self.tasks_q = tasks_q
        self.results_q = results_q

    def run(self):
        while True:
            image = self.tasks_q.get()
            # Do computations on image
            time.sleep(1)
            # Display the result on stream
            self.results_q.put("text")

# Tasks queue with size 1 - only want one image queued
# for processing.
# Queue size should therefore match number of processes
tasks_q, results_q = multiprocessing.Queue(1), multiprocessing.Queue()
processor = ImageProcessor(tasks_q, results_q)
processor.start()

def capture_display_video(vc):
    rval, frame = vc.read()
    while rval:
        if not tasks_q.full():
            tasks_q.put(frame)
        if not results_q.empty():
            text = results_q.get()
            cv2.putText(frame, text,(100, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 10, (225, 0, 0), 10)
        cv2.imshow("preview", frame)
        rval, frame = vc.read()

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
if not vc.isOpened():
    raise Exception("Cannot capture video")

capture_display_video(vc)
processor.terminate()