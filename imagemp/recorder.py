from imagemp import Consumer
import cv2

class Recorder(Consumer):
    def init_custom(self, *args, **kwargs):
        self.next_to_acquire = 'next'
        self.recorder = None
        self.filename = kwargs['filename']
        self.fourcc = kwargs['fourcc'] if 'fourcc' in kwargs else 'XVID'
        self.is_color = kwargs['is_color'] if 'is_color' in kwargs else True
        self.fps = kwargs['fps'] if 'fps' in kwargs else 30
        self.im_shape = kwargs['im_shape'] if 'im_shape' in kwargs else (2,2)
        if len(self.im_shape) > 2:
            print('Recorder: The image shape is {}, however only 2 dimensions will be used'.format(len(self.im_shape)))

    def run_pre_loop(self):
        cv2_fourcc = cv2.VideoWriter_fourcc(*self.fourcc)  # cv2.VideoWriter_fourcc() does not exist
        self.recorder = cv2.VideoWriter()
        ret = self.recorder.open(self.filename, cv2_fourcc, self.fps, self.im_shape[:2], isColor=self.is_color)
        if not ret:
            raise NameError("OpenCV recorder didn't open")

    def run_loop_pre_acquire(self):
        pass

    def run_loop_post_acquire(self):
        self.recorder.write(self.im.astype('uint8'))

    def close(self):
        self.recorder.release()
