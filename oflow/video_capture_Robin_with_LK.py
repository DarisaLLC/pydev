import cv2
from lucas_kanade_class0 import LucasKanade
import traceback


class MyVideo(LucasKanade):
    def __init__(self, videocolor, videolabel):
        super().__init__()
        self.video_color = videocolor
        self.video_label = videolabel
        self.cap = self.create_video()

    @staticmethod
    def create_video():
        cap = cv2.VideoCapture("/Users/arman/PycharmProjects/pydev/projects/wiic/video/2019_11_11/fiducial_straight_1.mp4")
#        cap.open(0)
        return cap

    def run(self):
        try:
            while True:
                # Capture frame-by-frame
                ret, frame = self.cap.read()
                if frame is None:
                    break

                # Display the resulting frame"
                # gray = cv2.cvtColor(frame, self.video_color)

                gray = self.frame_update(frame)

                cv2.putText(gray, self.video_label, (500, 500), cv2.FONT_HERSHEY_COMPLEX_SMALL, 10, (225, 0, 0), 10)
                cv2.imshow('frame', gray)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(e)
            traceback.print_exc()

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    video = MyVideo(cv2.COLOR_BGR2HSV, 'robin')
    video.run()
    video.close()
