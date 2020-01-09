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
        cap = cv2.VideoCapture("/Volumes/medvedev/_SP/2019_11_11/fiducial_straight_1.mp4")
#        cap.open(0)
        return cap

    def run(self):
        count = 0
        try:
            while True:
                # Capture frame-by-frame
                ret, frame = self.cap.read()
                if frame is None:
                    break

                gray = self.frame_update(frame)
                result_image_name = '/Users/arman/tmp/fid/fid' + str(count) + ".png"
                cv2.imwrite(result_image_name, frame)
                count = count+1
                
                cv2.putText(frame, self.video_label, (500, 500), cv2.FONT_HERSHEY_COMPLEX_SMALL, 10, (225, 0, 0), 10)
                cv2.imshow('frame', frame)
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
