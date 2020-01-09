import numpy as np
import cv2


class LucasKanade(object):
    def __init__(self):
        self.feature_params = self.ShiTomasi_corner_detection()
        self.lk_params = self.parameters_for_lucas_kanade_optical_flow()
        self.old_frame = None
        # Create some random colors
        self.dot_colors = np.random.randint(0, 255, (100, 3))
        self.p0 = self.p1 = None
        self.mask = np.zeros((1, 1))

    @staticmethod
    def ShiTomasi_corner_detection():
        # params for ShiTomasi corner detection
        feature_params = dict(maxCorners=100,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)
        return feature_params

    @staticmethod
    def parameters_for_lucas_kanade_optical_flow():
        # Parameters for Lucas-Kanade optical flow
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        return lk_params

    def frame_update(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.old_frame is not None:
            # calculate optical flow
            self.p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_frame, frame, self.p0, None, **self.lk_params)

            # Select good points
            good_new = self.p1[st == 1]
            good_old = self.p0[st == 1]

            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                self.mask = cv2.line(self.mask, (a, b), (c, d), self.dot_colors[i].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 5, self.dot_colors[i].tolist(), -1)

            # Now update the previous frame and previous points
            self.old_frame = frame.copy()
            self.p0 = good_new.reshape(-1, 1, 2)
            img = cv2.add(frame, self.mask)
        else:
            self.old_frame = frame
            self.p0 = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)

            # Create a self.mask image for drawing purposes
            self.mask = np.zeros_like(frame)
            img = frame
        return img


if __name__ == '__main__':
    obj = LucasKanade()
