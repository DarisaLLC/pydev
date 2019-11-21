
from imagemp import Consumer
from imagemp.process_runners.timers import ElapsedTimer
import cv2
import time

class SimpleDisplay(Consumer):
    def init_custom(self, *args, **kwargs):
        self.next_to_acquire = 'last'
        self.nframes_shown = 0
        self.t_acq = ElapsedTimer(moving_ave_n=10)
        self.t_fps = ElapsedTimer(moving_ave_n=10)
        self.t_fps.tic()

    def run_pre_loop(self):
        pass

    def run_loop_pre_acquire(self):
        self.nframes_shown += 1
        self.t_acq.tic()

    def run_loop_post_acquire(self):
        self.t_acq.toc()
        self.t_fps.toc()
        self.t_fps.tic()
        shape = self.im.shape
        self.im = self.im.astype('uint8')
        cv2.putText(self.im, 'frame#:   {}, update fps: {}, d(iframe): {}'.
                    format(int(self.last_timestamp),
                           round(10 / (self.t_fps.dts.moving_average+1e-10))/10, self.frame),
#                           self.shared_data.last_written_element.iframe - self.iframe),
                    (15, shape[0]-250), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        cv2.imshow('im', self.im)
        keypressed = cv2.waitKey(1)
        if keypressed in [81, 113]:  # 'Q' or 'q' : exit
            self.exit()
        print(' Display time: {}, Index: {}'.format(self.last_timestamp,self.iframe))

    def close(self):
        cv2.destroyAllWindows()
