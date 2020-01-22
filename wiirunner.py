import sys
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import math
from wiitricity_settings import video_rois, initialize_settings_from_video_roi, initialize_settings, region_of_interest
from common import anorm2, draw_str

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


def bgrFromHue(degrees):
    hsv = np.zeros((1, 1, 3), np.uint8)
    hsv[0, 0, 0] = ((degrees % 180) * 256) / 180.0
    hsv[0, 0, 1] = 255
    hsv[0, 0, 2] = 255
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
    tp = tuple([int(x) for x in bgr[0, 0, :]])
    return tp

def get_line_angle(line):
    x1, y1, x2, y2 = np.array(line, dtype=np.float64)
    radians = np.arctan2(y2 - y1, x2 - x1)
    return radians

def circular_mean(weights, angles):
    x = y = 0.
    for angle, weight in zip(angles, weights):
        x += math.cos(math.radians(angle)) * weight
        y += math.sin(math.radians(angle)) * weight

    mean = math.degrees(math.atan2(y, x))
    return mean

@unique
class State(Enum):
    eRight=-1,
    eLeft=+1,
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
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.frame_idx = 0
        self.prev_gray_frame = None
        self.prev_frame = None
        self.prev_angle = None
        self.prev_time = None
        self.current_angle = None
        self.current_time = None
        self.mask = None
        self.cycles = 0
        self.angle_diff_threshold = 5
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
        self.show_vectors = True
        self.show_axis = False

    def is_loaded(self):
        return self._is_loaded

    def is_valid(self):
        return self._is_valid

    def run(self):
        while True:
            ret, captured_frame = self.cap.read()
            if not ret: break

            gray_frame, frame = self.prepare_frame(captured_frame)
            self.get_flow(gray_frame)
            self.prev_frame = frame.copy()
            self.prev_gray_frame = gray_frame.copy()
            self.frame_idx = self.frame_idx + 1

            cv.namedWindow('Display', cv.WINDOW_NORMAL)
            cv.imshow("Display", self.display)
            ch = cv.waitKey(1)
            if ch == 27:
                break

    def prepare_frame(self, frame):
        frame = self.get_roi(frame)
        self.mask = region_of_interest(frame)
        self.display = frame.copy()
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        return gray_frame, frame

    def get_roi(self, frame):
        assert (self._is_valid)
        return frame[self.row_range[0]:self.row_range[1], self.column_range[0]:self.column_range[1]]



    ## left is positive angle difference, right is negative angle difference
    def update_direction(self):
        def set_state(_instant_angle):
            leaning_left = _instant_angle > (self.prev_angle + self.angle_diff_threshold)
            leaning_right = _instant_angle < (self.prev_angle - self.angle_diff_threshold)
            leaning_straight = (not leaning_left) and (not leaning_right)

            if self.current_state == State.eUnknown or self.current_state == State.eSame:
                if leaning_right: self.current_state = State.eRight
                elif leaning_left: self.current_state = State.eLeft
                else: self.current_state = State.eSame

            elif self.current_state == State.eRight:
                if leaning_right or leaning_straight:
                    self.current_state == State.eRight
                elif leaning_left:
                    self.current_state = State.eSame

            elif self.current_state == State.eLeft:
                if leaning_left or leaning_straight:
                    self.current_state == State.eLeft
                elif leaning_right:
                    self.current_state = State.eSame
            self.cycles += 1

            self.logger.info(('cycle, time, frame, direction:', self.cycles, self.current_time - self.prev_time,
                              instant_angle, self.current_state, leaning_left, leaning_right ))

        avg_angle = self.measure_tracks(self.keypoints, self.new_points)
        avg_angle = math.degrees(avg_angle) % 360
        self.direction.append(avg_angle)
        instant_angle = int(np.average(self.direction))
        if self.prev_angle is None:
            self.prev_angle = instant_angle
            self.prev_state = State.eUnknown
            self.prev_time = self.frame_idx
        else:
            self.current_angle = instant_angle
            self.current_time = self.frame_idx
            set_state(instant_angle)

            self.prev_angle = self.current_angle
            self.prev_state = self.current_state
            self.prev_time = self.current_time

 

    def get_flow(self, frame_gray):
        if len(self.tracks) > 0:
            img0, img1 = self.prev_gray_frame, frame_gray
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.settings['lk_params'])
            p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.settings['lk_params'])
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1
            new_tracks = []
            for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                tr.append((x, y))
                if len(tr) > self.track_len:
                    del tr[0]
                new_tracks.append(tr)
                cv.circle(self.display, (x, y), 2, (0, 255, 0), -1)
            self.tracks = new_tracks
            angles = []
            mags = []
            for track in self.tracks:
                a = track[0][0]
                b = track[0][1]
                c = track[1][0]
                d = track[1][1]
                line = [c, d, a, b]
                angle = get_line_angle(line)
                mag = math.sqrt((a - c) * (a - c) + (b - d) * (b - d))
                angles.append(angle)
                mags.append(mag)
            angles = np.asanyarray(angles)
            mags = np.asanyarray(mags)
            mean_angle = circular_mean(mags, angles)
            if mean_angle < 0:
                mean_angle = math.pi + math.pi + mean_angle
            mean_angle = mean_angle % math.pi / 2
            degrees = math.degrees(mean_angle)
            draw_str(self.display, (200, 200), 'Mean Angle: %d' % degrees)
            cv.polylines(self.display, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
            draw_str(self.display, (20, 20), 'track count: %d' % len(self.tracks))

        if self.frame_idx % self.detect_interval == 0:
            mask = self.mask.copy()
            for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                cv.circle(mask, (x, y), 5, 0, -1)
            p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **self.settings['feature_params'])
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    self.tracks.append([(x, y)])



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
        if self.prev_state != State.eUnknown and self.current_state != State.eUnknown and self.cycles > 1:
            direction = self.angle_states[self.current_state]
            cv.putText(self.display, direction, (self.width // 2, self.height // 2), cv.FONT_HERSHEY_DUPLEX,
                       2, (225, 0, 0), 7)

    def draw_tracks(self, points1, points2):
        if self.show_vectors:
            for i, (new, old) in enumerate(zip(points1, points2)):
                a, b = new.ravel()
                c, d = old.ravel()
                angle = get_line_angle([c, d, a, b])
                cl = bgrFromHue(math.degrees(angle))
                if a < self.width / 2: cl = (255,0,0)
                frame = cv.line(self.display, (c, d), (a, b), cl, 1)
                cv.circle(self.display, (c,d), 5, cl, 0, 3)

        self.draw_direction()
        cv.putText(self.display, str(self.frame_idx), (self.width // 16, (15*self.height) // 16), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 255, 0), 2)

    def measure_tracks(self):
        angles = []
        mags = []
        for track in self.tracks:
            a = track[0][0]
            b = track[0][1]
            c = track[1][0]
            d = track[1][1]
            line = [c, d, a, b]
            angle = get_line_angle(line)
            mag = math.sqrt((a - c) * (a - c) + (b - d) * (b - d))
            angles.append(angle)
            mags.append(mag)
        angles = np.asanyarray(angles)
        mags = np.asanyarray(mags)
        mean_angle = circular_mean(mags, angles)
        if mean_angle < 0:
            mean_angle = math.pi + math.pi + mean_angle

        degrees = math.degrees(mean_angle)
        draw_str(self.display, (200, 200), 'Mean Angle: %d' % degrees)
        return mean_angle % (2 * math.pi)




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
