from imagemp import Consumer
import numpy as np
import cv2
#from tracker.findfish4class import FindFish4


class TrackPad(Consumer):
    def init_custom(self, *args, **kwargs):
        self.next_to_acquire = 'next'
        self.vid_av_filename = kwargs['vid_av_filename'] if 'vid_av_filename' in kwargs else None
#        self.im_av = np.load(self.vid_av_filename)
#        self.fish_tracker = FindFish4(im_av=self.im_av)

    def run_pre_loop(self):
        pass

    def run_loop_pre_acquire(self):
        pass

    def run_loop_post_acquire(self):
        shape = self.im.shape
        new_im = self.im.astype('uint8')
        timestamp = self.last_timestamp
        iframe = self.iframe
        self.im = self.im.astype('uint8')
        cv2.putText(new_im,' Analysis time: {}, Index: {}'.format(self.last_timestamp,self.iframe),
                    (15, shape[0] - 350), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

    def close(self):
        pass

