import sys
import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import math
from padChecker import padChecker
from wiitricity_settings import video_rois, initialize_settings_from_video_roi, initialize_settings
from wiitricity_settings import region_of_interest, vertical, _draw_str, get_logger, get_line_angle
from wiitricity_settings import circular_mean, line_yhist, compute_lines, quadrilateral_detection
from eval_homography import image_change
from skimage.feature import peak_local_max
from skimage.util import img_as_float, img_as_ubyte


import numpy as np

import logging
from datetime import datetime


class gpad_odometry:

    def __init__(self, video_source_info, user_roi, data_path):
        self.source_info = video_source_info
        self.data_cache = data_path
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
        self.abs_diff = None
        self.prev_time = None
        self.current_angle = None
        self.current_distance = None
        self.current_time = None
        self.prev_angle = None
        self.prev_distance = None
        self.prev_time = None
        self.flow_mask = None
        self.mask = None
        self.cycles = 0
        self.angle_diff_threshold = 5
        self.logger = get_logger()
        self.logger.info('Source is ' + self.source)
        video_roi = user_roi
        if self.source == 'Camera':
            camera_roi = dict(row_low=0, row_high=int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)) - 1,
                              column_low=0, column_high=int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH)) - 1)
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
        self.show_vectors = False
        self.show_pads = False
        self.show_ga_score = True

        # Get a GA pad checker object
        self.checker = padChecker(self.data_cache)
        self.ga_results = None
        self.prev_ga_results = None
        self.prev_lines = None
        self.lines = None

    def distance_is_available(self):
        return not (self.current_distance is None)

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
            self.line_processing(gray_frame)
            #     self.display = self.detect_ga(frame)
            self.prev_frame = frame.copy()
            self.prev_gray_frame = gray_frame.copy()
            self.frame_idx = self.frame_idx + 1

            self.draw_direction()
            self.draw_frame_info()
            cv.namedWindow('Display', cv.WINDOW_NORMAL)
            cv.imshow("Display", self.display)
            ch = cv.waitKey(1)
            if ch == 27:
                break
            elif ch == ord("v"):
                self.show_vectors = not self.show_vectors

    def prepare_frame(self, frame):
        frame = self.get_roi(frame)
        self.mask = region_of_interest(frame)
        self.flow_mask = vertical(frame)
        self.display = frame.copy()
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # filename = '/Users/arman/tmpin/' + str(self.frame_idx) + '.png'
        # cv.imwrite(filename, frame)
        return gray_frame, frame

    def line_processing(self, gray_frame):
        half_width = self.width//2
        half_height = self.height//2

        topleft = (half_width//8, half_height//4)
        botright = half_width - topleft[0], half_height - topleft[1]
        lines_valid_region = (topleft, botright)
        lines_mask_region = [(0,0),(half_width-1, half_height//3)]
        half_res = cv.resize(gray_frame,(half_width,half_height))
        half_res = cv.GaussianBlur(half_res, (0,0), 1)

        (lines, locations, directions, strengths, kde) = compute_lines(half_res, lines_valid_region,(15, 100))
        print((len(locations), ' Lines '))
       # parallels = quadrilateral_detection(lines, locations, directions, strengths, kde, (15,100))


        if self.prev_gray_frame is None:
            self.prev_gray_frame = half_res
            self.prev_lines = lines
        else:
            self.prev_lines = self.lines
            self.lines = lines

        def bgrFromHue(degrees):
            hsv = np.zeros((1, 1, 3), np.uint8)
            hsv[0, 0, 0] = ((degrees % 360) * 255) / 360.0
            hsv[0, 0, 1] = 255
            hsv[0, 0, 2] = 255
            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            tp = tuple([int(x) for x in bgr[0, 0, :]])
            return tp

        # if not (self.prev_lines is None):
        #     for line in self.prev_lines:
        #         x1, y1, x2, y2 = line
        #         cv.line(self.display, (x1, y1), (x2, y2), (0, 0, 255), 2)

        vornoi = np.zeros((self.height, self.width, 1), dtype="uint8")
        vornoi += 255

        if not (self.lines is None):
            for line in self.lines:
                x1, y1, x2, y2 = line
                radians = np.arctan2(y2 - y1, x2 - x1)
                cl = bgrFromHue(math.degrees(radians))
                cv.line(self.display, (x1+x1, y1+y1), (x2+x2, y2+y2), cl, 2)
                cv.line(vornoi, (x1+x1, y1+y1), (x2+x2, y2+y2), (0,0,0), 2)

        dist_transform = cv.distanceTransform(vornoi, cv.DIST_L2, 5, None, cv.CV_32F)

        ## Find all local maximas in the distance image
        peaks = peak_local_max(img_as_float(dist_transform),min_distance=50, indices=False)
        peaks_img = img_as_ubyte(peaks)
        for y in range(0, self.height):
            for x in range(0, self.width):
                # threshold the pixel
                if peaks_img[y,x]:
                    cv.circle(self.display, (x,y), 30, (0,0,255), 5)

        cv.imwrite('/Users/arman/vtmp/' + str(self.frame_idx) + '.png', dist_transform)
        cv.imwrite('/Users/arman/vvtmp/' + str(self.frame_idx) + '.png', vornoi)
        cv.imwrite('/Users/arman/vvvtmp/' + str(self.frame_idx) + '.png', peaks_img)

        # dist2 = cv.normalize(dist_transform, None, 255, 0, cv.NORM_MINMAX, cv.CV_8UC1)
        # cv.namedWindow('Display', cv.WINDOW_NORMAL)
        # cv.imshow("Display", dist2)
        # ch = cv.waitKey(10)


    def detect_ga(self, frame):
        res = self.display
        #    if self.distance_is_available() and int(self.current_distance) < 5:
        # readout,thresh, seethrough, cnts, hulls
        checkout = self.checker.check(frame, self.mask)
        if self.ga_results is None:
            self.ga_results = checkout
        else:
            self.prev_ga_result = self.ga_results
            self.ga_results = checkout

        if not (self.ga_results is None):
            score = str(self.ga_results[0])
            _draw_str(self.display, (200, 400), score, 2.0)
            #  res = cv.drawContours(res, self.ga_results[4], -1, (255, 0, 0), 3)
            if self.show_pads:
                res = np.vstack((res, self.ga_results[2]))
        return res

    def get_roi(self, frame):
        assert (self._is_valid)
        return frame[self.row_range[0]:self.row_range[1], self.column_range[0]:self.column_range[1]]

    def get_flow(self, frame_gray):
        if len(self.tracks) > 0:
            img0, img1 = self.prev_gray_frame, frame_gray

            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.settings['lk_params'])
            p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.settings['lk_params'])
            d = abs(p0 - p0r).reshape(-1, 2).max(-1)
            good = d < 1
            new_tracks = []
            for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                tr.append((x, y))
                if len(tr) > self.track_len:
                    del tr[0]
                new_tracks.append(tr)
                if self.show_vectors:
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
            mean_angle = mean_angle % (2 * math.pi)
            mean_mags = np.mean(mags)

            if self.prev_angle is None:
                self.prev_angle = mean_angle
            if self.prev_distance is None:
                self.prev_distance = mean_mags

            self.prev_distance = self.current_distance
            self.current_distance = mean_mags

            if int(self.current_distance) > 0:
                self.prev_angle = self.current_angle
                self.current_angle = mean_angle
            else:
                self.current_angle = self.prev_angle

            if self.show_vectors:
                cv.polylines(self.display, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                _draw_str(self.display, (20, self.height - 100), ' %d' % len(self.tracks), 2.0)

        if self.frame_idx % self.detect_interval == 0:
            mask = self.flow_mask.copy()
            for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                cv.circle(mask, (x, y), 5, 0, -1)
            p = cv.goodFeaturesToTrack(frame_gray, mask=mask, **self.settings['feature_params'])
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
        if not (self.current_angle is None):
            _draw_str(self.display, (self.width // 8, (15 * self.height) // 16), 'Mean Direction: %d' % math.degrees(self.current_angle), 2.0)
        if not (self.current_distance is None):
            _draw_str(self.display, ((2*self.width) // 3, (15 * self.height) // 16), 'Mean Distance: %d' % self.current_distance, 2.0)

    def draw_frame_info(self):
        cv.putText(self.display, str(self.frame_idx), (self.width // 16, (15 * self.height) // 16),
                   cv.FONT_HERSHEY_SIMPLEX,
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

    if len(sys.argv) < 1:
        sys.exit(1)

    default_data_path = os.path.dirname(os.path.realpath(__file__)) + '/projects/wiic/'
    default_is_good = Path(default_data_path).exists() and Path(default_data_path).is_dir()
    msg = ''
    if len(sys.argv) == 2 and default_is_good:
        data_path = default_data_path
        print(('Default Data Directory is used ', default_data_path))
    elif len(sys.argv) == 3 and Path(sys.argv[2]).exists() and Path(sys.argv[2]).is_dir():
        data_path = sys.argv[2]
        print(('User Supplied Data Directory is used ', data_path))
    elif len(sys.argv) == 3 and default_is_good:
        data_path = default_data_path
        print(('User Supplied is Invalid: Default Data Directory is used ', default_data_path))
    else:
        print(('User Supplied Data Directory does not exist or is Invalid ', ' Default Data Directory does not exist '))
        sys.exit(1)

    # use video_roi['hd'] for newer video
    runner = gpad_odometry(sys.argv[1], video_rois['hd'], data_path)
    loaded = runner.is_loaded()
    if not loaded:
        self.logger.info('Video Did not Load')
        exit(1)
    runner.run()

    cv.destroyAllWindows()

#  cap.release()
#  cv.waitKey(0)
#
