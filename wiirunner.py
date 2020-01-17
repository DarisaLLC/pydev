import sys
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import math
from wiitricity_settings import video_rois, initialize_settings_from_video_roi, initialize_settings
import logging
from datetime import datetime
from enum import Enum, unique

# create logger with 'wiirunner'
def get_logger():
    logfilename = 'wiirunner'+datetime.now().strftime("-%d-%m-%Y_%I-%M-%S_%p")+'.log'
    logger = logging.getLogger('wiirunner')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(logfilename)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def get_line_angle(line):
    """Calculates the angle of a line.

    Args:
        line: Vector of x1, y1, x2, y2, i.e. two points on the line.

    Returns:
        Float, angle in degrees.
    """
    x1, y1, x2, y2 = np.array(line, dtype=np.float64)
    radians = np.arctan2(y2 - y1, x2 - x1)
    return radians


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

@unique
class State(Enum):
    eRight=1,
    eLeft=-1,
    eSame=0,
    eUnknown=-2

class movement_direction:

    def __init__(self, video_source_info, user_roi ):
        self.source_info = video_source_info
        self.source = None
        if self.source_info == '0' or self.source_info == '1':
            self.source = 'Camera'
        elif isinstance(self.source_info, str) and Path(self.source_info).exists() and Path(sys.argv[1]).is_file():
            self.source = self.source_info
        self._is_valid = not (self.source is None)

        if self.source == 'Camera':
            self.cap = cv.VideoCapture(int(self.source_info))
        else:
            self.cap = cv.VideoCapture(self.source_info)
        self._is_loaded = True
        self.frame_count = 0
        self.prev_frame = None
        self.prev_angle = None
        self.current_angle = None
        self.angle_estimate_available = False
        self.angle_posts = dict(Right=75,Left=120)
        self.angle_diff_threshold = 7
        self.current_state = State.eUnknown
        self.prev_state = self.current_state
        self.logger = get_logger()
        self.logger.info('Source is ' + self.source)
        self.angle_states = {}
        self.angle_states[State.eRight] = 'Right'
        self.angle_states[State.eLeft] = 'Left'
        self.angle_states[State.eSame] = 'Straight'
        self.angle_states[State.eUnknown] = 'Unknown'
        video_roi = user_roi
        if self.source == 'Camera':
            camera_roi = dict(row_low=0, row_high=int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))-1,
                                column_low=0, column_high=int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))-1)
            video_roi = camera_roi

        self.logger.info(video_roi)

        self.row_range = (video_roi['row_low'], video_roi['row_high'])
        self.column_range = (video_roi['column_low'], video_roi['column_high'])
        self.settings = initialize_settings_from_video_roi(video_roi)
        self.width = video_roi['column_high'] - video_roi['column_low']
        self.height = video_roi['row_high'] - video_roi['row_low']

        assert (self.width <= int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH)))
        assert (self.height <= int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
        self.feature_params = self.settings['feature_params']
        self.lk_params = self.settings['lk_params']
        self.max_dist = self.settings['max_distance']
        self.prev_frame = None
        self.canvas = None
        self.keypoint_dist = 0
        self.min_features = self.settings['min_features']
        self.old_points = []
        self.new_points = []
        self.keypoints = []
        self.m0 = self.m1 = 0
        self.display = None
        self.debug = False
        self.direction = []
        self.avg_angle = None
        self.show_vectors = False
        self.show_axis = False

    def is_loaded(self):
        return self._is_loaded

    def is_valid(self):
        return self._is_valid

    def run(self):
        ret, captured_frame = self.cap.read()
        if not ret:
            return
        self.prev_frame = self.prepare_frame(captured_frame)
        self.logger.info(('frame: ', self.frame_count))
        self.frame_count = self.frame_count + 1
        cv.namedWindow('Frame', cv.WINDOW_NORMAL)

        while self.cap.isOpened():
            ret, captured_frame = self.cap.read()
            if not ret: break
            self.frame_count = self.frame_count + 1
            frame = self.prepare_frame(captured_frame)
            assert (not (self.prev_frame is None))
            self.update_features()
            self.get_flow(frame)
            self.prev_frame = frame.copy()

            self.draw_tracks( self.keypoints, self.new_points)
            if self.show_axis:
                cv.line(self.display, (0, self.height // 2), (self.width, self.height // 2), (255, 0, 0), 1)
                cv.line(self.display, (self.width // 2, 0), (self.width // 2, self.height), (255, 0, 0), 1)
            cv.imshow("Frame", self.display)

            key = cv.waitKey(1) & 0xFF

    def prepare_frame(self, frame):
        frame = self.get_roi(frame)
        self.display = frame.copy()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        return frame

    def get_roi(self, frame):
        assert (self._is_valid)
        return frame[self.row_range[0]:self.row_range[1], self.column_range[0]:self.column_range[1]]

    def get_features(self):
        self.keypoint_dist = 0
        # collect features on the prev frame and compute its geometric median
        self.old_points = cv.goodFeaturesToTrack(self.prev_frame, mask=None, **self.feature_params)
        self.keypoints = np.copy(self.old_points)

    def update_features(self):
        # if moved too much re create features on the prev frame
        if self.keypoint_dist > self.max_dist:
            self.get_features()

            def set_state(_instant_angle):
                if _instant_angle <= self.angle_posts['Right']:
                    self.current_state = State.eRight
                elif _instant_angle >= self.angle_posts['Left']:
                    self.current_state = State.eLeft
                else:
                    self.current_state = State.eSame

            if self.debug: self.logger.debug('max distance passed: reset ')
            avg_angle = self.measure_tracks(self.keypoints, self.new_points)
            avg_angle = math.degrees(avg_angle) % 360
            self.direction.append(avg_angle)
            instant_angle = int(np.average(self.direction))
            if self.prev_angle is None:
                self.prev_angle = instant_angle
                self.prev_state = State.eUnknown

            self.current_angle = instant_angle
            diff = self.current_angle - self.prev_angle
            ok = diff < self.angle_diff_threshold
            if ok and self.current_state == self.prev_state:
                self.current_state = State.eSame
            else:
                set_state(instant_angle)

            self.prev_angle = self.current_angle
            self.prev_state = self.current_state

            ## determine current direction:
            ## difference = current - prev
            ## if difference < 0 threshold and current is not Unknow ==> Straight
            ## if prev != current ==> current

            self.logger.info(('frame, direction:', self.frame_count, instant_angle, self.current_state ))


        # if few new points create features on the prev frame
        elif len(self.new_points) < self.min_features:
            self.get_features()
            if self.debug: self.logger.debug('copied old points to keypoints   ')
        else:
            # check number of features in each quadrant to ensure a good distribution of features across entire image
            nw = ne = sw = se = 0
            w = self.width
            h = self.height
            for x, y in self.new_points:
                if x > w // 2:
                    if y > h // 2:
                        se += 1
                    else:
                        ne += 1
                else:
                    if y > h // 2:
                        sw += 1
                    else:
                        nw += 1

            self.num_features = min((nw, ne))
            if self.num_features < self.min_features // 4:
                self.get_features()
        #        if self.debug: self.logger.debug('too few features reset  ')
            else:
                # just copy new points to old points
                dim = np.shape(self.new_points)
                self.old_points = np.reshape(self.new_points, (-1, 1, 2))
                if self.debug: self.logger.debug('ok')

    def get_flow(self, frame):
        self.new_points, self.status, self.error = cv.calcOpticalFlowPyrLK(self.prev_frame, frame, self.old_points,
                                                                           None, **self.lk_params)
        self.keypoints = np.reshape(self.keypoints, (-1, 1, 2))  # TODO find out why this is necessary?!
        self.old_points = self.old_points[self.status == 1]
        self.new_points = self.new_points[self.status == 1]
        self.keypoints = self.keypoints[self.status == 1]
        self.keypoint_dist += self.get_mean_distance_2d(self.old_points, self.new_points)

    def get_mean_distance_2d(self, features1, features2):
        num_features = min((len(features1), len(features2)))
        features1 = np.reshape(features1, (num_features, 2))
        features2 = np.reshape(features2, (num_features, 2))

        features = zip(features1, features2)
        n = 0
        dist = 0
        for f1, f2 in features:
            dx = f1[0] - f2[0]
            dy = f1[1] - f2[1]
            d = math.sqrt(dx ** 2 + dy ** 2)
            dist += d
            n += 1

        if n == 0:
            return 0

        dist /= n
        return dist

    def draw_direction(self):
        if self.prev_state != State.eUnknown and self.current_state != State.eUnknown:
            direction = self.angle_states[self.current_state]
            cv.putText(self.display, direction, (self.width // 2, self.height // 2), cv.FONT_HERSHEY_DUPLEX,
                       2, (225, 0, 0), 7)

    def draw_tracks(self, points1, points2):
        if self.show_vectors:
            for i, (new, old) in enumerate(zip(points1, points2)):
                a, b = new.ravel()
                c, d = old.ravel()
                cl = (0,255,0)
                if a < self.width / 2: cl = (255,0,0)
                frame = cv.line(self.display, (c, d), (a, b), cl, 1)
                cv.circle(self.display, (a,b), 5, cl, 0, 3)

        self.draw_direction()
        cv.putText(self.display, str(self.frame_count), (self.width // 16, (15*self.height) // 16), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 255, 0), 2)

    def measure_tracks(self, points1, points2):
        angles = []
        for i, (new, old) in enumerate(zip(points1, points2)):
            a, b = new.ravel()
            c, d = old.ravel()
            line = [c, d, a, b]
            angle = get_line_angle(line)
            angles.append(angle)

        angles = np.asanyarray(angles)
        mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
        if mean_angle < 0:
            mean_angle = math.pi + math.pi + mean_angle

        return mean_angle




if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    # use video_roi['hd'] for newer video
    runner = movement_direction(sys.argv[1], video_rois['hd'])
    loaded = runner.is_loaded()
    if not loaded:
        self.logger.info('Video Did not Load')
        exit(1)
    runner.run()

#  cv.waitKey(0)
#  cv.destroyAllWindows()
#  cap.release()
