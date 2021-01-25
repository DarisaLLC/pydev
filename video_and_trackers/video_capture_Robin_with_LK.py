import cv2
from lucas_kanade_class0 import LucasKanade
import traceback


def get_line_angle(line):
    """Calculates the angle of a line.

    Args:
        line: Vector of x1, y1, x2, y2, i.e. two points on the line.

    Returns:
        Float, angle in degrees.
    """
    x1, y1, x2, y2 = np.array(line, dtype=np.float64)
    radians = np.arctan2(y2 - y1, x2 - x1)
    return math.degrees(radians) % 360

def point_to_point_dist(point_a, point_b):
    """Finds euclidean distance between two points.

    Args:
        point_a: Tuple (x, y) point.
        point_b: Tuple (x, y) point.

    Returns:
        Float, distance.
    """
    x1, y1 = np.array(point_a, dtype=np.float64)
    x2, y2 = np.array(point_b, dtype=np.float64)
    return math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

class MyVideo(LucasKanade):
    def __init__(self, videocolor, videolabel):
        super().__init__()
        self.video_color = videocolor
        self.video_label = videolabel
        self.cap = self.create_video()

    @staticmethod
    def create_video():
     #   cap = cv2.VideoCapture("/Users/arman/PycharmProjects/pydev/projects/wiic/video/2019_11_11/fiducial_straight_1.mp4")
        cap = cv2.VideoCapture(0)
        cap.open(0)
        return cap

    def run(self):
        try:
            while True:
                # Capture frame-by-frame
                ret, frame = self.cap.read()
                if frame is None: continue
#                    raise NameError('None frame')

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
    video = MyVideo(cv2.COLOR_BGR2HSV, '+++')
    video.run()
    video.close()
