import sys
import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import cv2
import math
from padChecker import padChecker
from wiitricity_settings import video_rois, initialize_settings_from_video_roi, initialize_settings
from wiitricity_settings import region_of_interest, vertical, _draw_str, get_logger, get_line_angle
from wiitricity_settings import circular_mean, compute_lines, filter_lines
from wiitricity_utils import roiPts
import dlib

import geometry as utils
from skimage.feature import peak_local_max
from skimage.util import img_as_float, img_as_ubyte
import numpy as np
import time


import logging
from datetime import datetime

DEFAULT_FILTER = lambda x1, y1, x2, y2: abs(atan2(y2 - y1, x2 - x1) * 180.0 / np.pi - 0) < 5 or abs(
    atan2(y2 - y1, x2 - x1) * 180.0 / np.pi - 180) < 5

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
            self.cap = cv2.VideoCapture(int(self.source_info))
        else:
            self.cap = cv2.VideoCapture(self.source_info)
        self._is_loaded = True
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.frame_idx = 0
        self.prev_channel = None
        self.prev_frame = None
        self.abs_diff = None
        self.current_angle = None
        self.current_distance = None
        self.current_speed = None
        self.prev_angle = None
        self.prev_distance = None
        self.prev_speed = None
        self.flow_mask = None
        self.mask = None
        self.cycles = 0
        self.angle_diff_threshold = 5
        self.logger = get_logger()
        self.logger.info('Source is ' + self.source)
        video_roi = user_roi
        if self.source == 'Camera':
            camera_roi = dict(row_low=0, row_high=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) - 1,
                              column_low=0, column_high=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) - 1)
            video_roi = camera_roi

        self.logger.info(video_roi)

   
        self.settings = initialize_settings_from_video_roi(video_roi)
        self.reduction = self.settings['reduction']
        reduction = self.reduction
        self.row_range = (video_roi['row_low']//reduction, video_roi['row_high']//reduction)
        self.column_range = (video_roi['column_low']//reduction, video_roi['column_high']//reduction)
        self.width = self.column_range[1] - self.column_range[0] + 1
        self.height = self.row_range[1] - self.row_range[0] + 1
        self.image_bounds = np.array([[self.column_range[0], self.row_range[0]],
                                      [self.column_range[0] + self.width, self.row_range[0]],
                                      [self.column_range[0] + self.width, self.row_range[0] + self.height],
                                      [self.column_range[0], self.row_range[0] + self.height]])

        cap_size = (self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)/self.reduction, self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/self.reduction)
        if cap_size[0] < self.width or cap_size[1] < self.height:
            print((' Error \n Mismatch of Video Source and Region Specification \n Check Region Information '
                   + str(self.height) + ' <= ' + str(cap_size[1])
                   + ' or '  + str(self.width) + ' <= ' + str(cap_size[0])))
            sys.exit(1)
            
        assert (self.width <= int(cap_size[0]))
        assert (self.height <= int(cap_size[1]))
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
        self.show_lineout_lines = False
        self.show_lineout_rects = False
        self.gray_out_mask = None
        self.line_processing_enabled = False
        
        # Get a GA pad checker object
        self.checker = padChecker(self.data_cache)
        self.ga_results = None
        self.prev_ga_results = None
        self.prev_lines = None
        self.lines = None
        self.kdtree = None
        
        # Tracker
        # construct a dlib tracker
        self.tracker = dlib.correlation_tracker()
        self.tracker_started = False
        self.tracker_running = False
        
        self.vertical_horizon = self.height * self.settings['vertical_horizon_norm']

    def distance_is_available(self):
        return not (self.current_distance is None)

    def is_loaded(self):
        return self._is_loaded

    def is_valid(self):
        return self._is_valid

    def run(self):
        run_vpd = True
        
        while True:
            ret, captured_frame = self.cap.read()
            if not ret: break
            channel_in, frame = self.prepare_frame(captured_frame)

            self.get_flow(channel_in)
            self.display = self.detect_ga(frame)
            self.track_ga(frame)
            self.line_processing(channel_in)

            self.prev_frame = frame.copy()
            self.prev_channel = channel_in.copy()
            self.frame_idx = self.frame_idx + 1

            self.draw_direction()
            self.draw_frame_info()
            cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
            cv2.imshow("Display", self.display)

            wait = 0 if self.settings['display_click_after_frame'] == True else self.settings['display_frame_delay_seconds']
            ch = cv2.waitKey(wait)
            if ch == 27:
                break
            elif ch == ord("v"):
                self.show_vectors = not self.show_vectors
            elif ch == ord("h"):
                self.settings['display_source'] = 'hue'
            elif ch == ord("g"):
                self.settings['display_source'] = 'gray'
            elif ch == ord("c"):
                self.settings['display_source'] = 'native_color'

    def synthesize_frame(self):
        cols = self.width
        rows = self.height
        frame = np.zeros((rows,cols, 3), dtype=np.uint8)
        left = 100
        right = 300
        rect = np.array([[
            (left, 50), (right, 50), (223,63), (200, 63)]], np.int32)
        rect2 = rect + [300,0]
        cv2.fillConvexPoly(frame, rect, (128,128,128))
        cv2.fillConvexPoly(frame, rect2, (192,192,192))
        
        
        return frame
    
        
    def prepare_frame(self, input_frame):
        reduction = self.settings['reduction']
        h,w,c = input_frame.shape
        new_size = (w//self.reduction,h//self.reduction)
        mframe = cv2.medianBlur(input_frame, self.settings['ppMedianBlur'])
        frame = cv2.resize(mframe, new_size, cv2.INTER_AREA)
        if self.settings['synthesize_test']:
            frame = self.synthesize_frame ()
        frame = self.get_roi(frame)
        self.mask = region_of_interest(frame)
        self.flow_mask = vertical(frame)
        # Choose display source
        if self.settings['display_source'] == 'gray':
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.display = cv2.merge([g,g,g])
        elif self.settings['display_source'] == 'hue':
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
            h, s, v = cv2.split(frame)
            self.display = cv2.merge([h,h,h])
        else:
            self.display = frame.copy()
            
        ## Choose the channel_in to use
        channel_in = None
        
        if self.settings['use_channel'] == 'gray':
            channel_in = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif self.settings['use_channel'] == 'hsv':
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
            channel_in, s, v = cv2.split(frame)
            
        else: assert(False)


        if not (self.settings['write_frames_path'] is None):
            filename = '/Users/arman/tmpin/' + str(self.frame_idx) + '.png'
            cv2.imwrite(filename, frame)

        return channel_in, frame

    def detect_ga(self, frame):
        res = self.display
        '''
        h_score,thresh, res, scores, hulls, rects, order
		'''
        
        checkout = self.checker.check(frame, self.mask)
        rects = checkout[5]
        scores = checkout[3]
        order = checkout[6]
        if self.ga_results is None:
            self.ga_results = checkout
        else:
            self.prev_ga_result = self.ga_results
            self.ga_results = checkout
    
        if len(order) > 0:
            score = str(scores[order[0]])
            self.draw_rectangle(rects[order[0]], line_color = (0, 255, 128), fill_color = (0, 255, 0), opacity = 0.3)

            self.logger.info('GA Detection: ' + str(score))
            
            if not self.tracker_started:
                # @todo put first, 2nd and third line in roiPts
                points = cv2.boxPoints(rects[0])
                tl, br = roiPts(points)
                (x, y, w, h) = (tl[0], tl[1], br[0] - tl[0], br[1] - tl[1])
                roi_pts = [x, y, x + w, y + h]
                self.tracker.start_track(frame, dlib.rectangle(*roi_pts))
                self.tracker_started = True
                self.logger.info('Tracker Initiated @: ' + str(x) + ',' + str(y) + ' width ' + str(w) +','+ str(h))
            
            if self.show_pads:
                res = np.vstack((res, self.ga_results[2]))
        return res

    def track_ga(self, frame):
        # @todo if newly segmented and tracked from the past overlap, restart tracker with the union of the two tracker
        if self.tracker_started:
            #grab the position of the tracked
            # object
            self.tracker.update(frame)
            pos = self.tracker.get_position()
            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            ctr = ((startX + endX) / 2, (startY + endY) / 2)
            size = (endX - startX, endY - startY)
            rect = (ctr, size, 0.0)
            # draw the bounding box from the correlation object tracker
            self.draw_rectangle(rect, line_color = (0, 255, 128), fill_color = (0, 255, 0), opacity = 0.0)
            cv2.putText(self.display, ' t ', (startX, startY - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            self.tracker_running = True
            self.logger.info('Tracker Updated @: ' + str(startX) + ',' + str(startY) + ' width ' + str(size[0]) + ',' + str(size[1]))
            
            
    def line_processing(self, channel_in):
        if self.line_processing_enabled:
            topleft = (self.width // 8, self.height // 4)
            botright = self.width - topleft[0], self.height - topleft[1]
            lines_valid_region = (topleft, botright)
            expected_minimum_size = self.settings['expected_minimum_size']
            view_angle = np.pi if self.current_angle is None else self.current_angle
            view_size = expected_minimum_size #if self.current_distance is None else expected_minimum_size + self.current_distance / 4
            # (rects, lines, directions, xc, yc, cands)
            lineouts  = compute_lines(channel_in, view_angle,  view_size, self.vertical_horizon,self.logger, self.settings)
            rects = lineouts[0]
            lines = lineouts[1]
            directions = lineouts[2]
            xc = lineouts[3]
            yc = lineouts[4]
            cands = lineouts[5]
            
            print((len(lines), ' Lines '))
            if self.prev_channel is None:
                self.prev_channel = channel_in
                self.prev_lines = lineouts
            else:
                self.prev_lines = self.lines
                self.lines = lineouts
    
            if self.vertical_horizon >= 0 and self.vertical_horizon < self.height:
                cv2.line(img=self.display, pt1=(0, int(self.vertical_horizon)),pt2=(self.width-1,int(self.vertical_horizon)),
                         color=(0,255,0,80), thickness=4, lineType=cv2.LINE_AA)
    
            if self.show_lineout_rects:
                self.draw_rectangles(rects)
            if self.show_lineout_lines:
                self.draw_segments(lines, directions, xc, yc, cands)

    #def reconcile():
       
       ## find intersections between ga rects and line processing rects
       

    def draw_segments(self, segments: list, directions: list = None, xc: list = None, yc: list = None, cands: list= None):
        """Draws the segments contained in the first parameter onto the base image passed as second parameter.

        This function displays the image using the third parameter as title.
        The indices associated to the segments are rendered on the image depending on 'render_indices'.
        A list of colors can be passed as argument to specify the colors to be used for different segment clusters.

        :param segments: List of segment clusters.
        :param base_image: Base image over which to render the segments.
        :param render_indices: Boolean flag indicating whether to render the segment indices or not.
        """
        base_image = self.display
        def bgrFromHue(degrees):
            hsv = np.zeros((1, 1, 3), np.uint8)
            hsv[0, 0, 0] = ((degrees % 90) * 255) / 90.0
            hsv[0, 0, 1] = 255
            hsv[0, 0, 2] = 255
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            tp = tuple([int(x) for x in bgr[0, 0, :]])
            return tp

        for segment_index, segment in enumerate(segments):
            p0, p1 = np.array([segment[0], segment[1]]), np.array([segment[2], segment[3]])
            angle = directions[segment_index]
            label = int(math.degrees(angle))
            cl = bgrFromHue(label)
            cv2.line(img=base_image, pt1=(int(segment[0]), int(segment[1])),
                     pt2=(int(segment[2]), int(segment[3])),
                     color=cl, thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(base_image, str(label), (int(segment[0]), int(segment[1])),
                        cv2.FONT_HERSHEY_PLAIN, 0.8,
                        cl, 1)

        if not (cands is None):
            for (index,cand) in enumerate(cands):
                i = cand[0]
                j = cand[1]
                cv2.line(img=base_image, pt1=(int(xc[i]), int(yc[i])),
                         pt2=(int(xc[j]), int(yc[j])), color=(255,255,255), thickness=1, lineType=cv2.LINE_AA)

    def draw_rectangles(self, rectangles: list, line_color = (0, 0, 255), fill_color = (255, 0, 0), opacity = 0.3):
        """Draws the rectangles contained in the first parameter onto the base image passed as second parameter.

		This function displays the image using the third parameter as title.

		:param rectangles: List of rectangles.
		:param base_image: Base image over which to render the rectangles.
		:param windows_name: Title to give to the rendered image.
		"""
        mask = np.zeros_like(self.display)
        for rectangle in rectangles:
            points = cv2.boxPoints(rectangle)
            cv2.polylines(self.display, np.int32([points]), True, line_color, 1, cv2.LINE_AA)
            cv2.fillConvexPoly(mask, np.int32([points]), fill_color, cv2.LINE_AA)
            cv2.addWeighted(self.display, 1, mask, opacity, 0, self.display)

    def draw_rectangle(self, rectangle: list, line_color = (0, 0, 255), fill_color = (255, 0, 0), opacity = 0.3):
        """Draws the rectangles contained in the first parameter onto the base image passed as second parameter.

		This function displays the image using the third parameter as title.

		:param rectangles: List of rectangles.
		:param base_image: Base image over which to render the rectangles.
		:param windows_name: Title to give to the rendered image.
		"""
        mask = np.zeros_like(self.display)
        points = cv2.boxPoints(rectangle)
        cv2.polylines(self.display, np.int32([points]), True, line_color, 1, cv2.LINE_AA)
        cv2.fillConvexPoly(mask, np.int32([points]), fill_color, cv2.LINE_AA)
        cv2.addWeighted(self.display, 1, mask, opacity, 0, self.display)

  

    def get_roi(self, frame):
        assert (self._is_valid)
        return frame[self.row_range[0]:self.row_range[1], self.column_range[0]:self.column_range[1]]


    def get_flow(self, frame_gray):
        mean_angle = np.pi
        mean_mags = 0
        
        if len(self.tracks) > 0:
            img0, img1 = self.prev_channel, frame_gray

            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.settings['lk_params'])
            p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.settings['lk_params'])
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
                    cv2.circle(self.display, (x, y), 2, (0, 255, 0), -1)
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
            self.prev_distance = 0

        current_dist = self.prev_distance + mean_mags
        self.prev_distance = self.current_distance
        self.current_distance = current_dist

        self.prev_speed = self.current_speed
        self.current_speed = mean_mags
        
        if int(self.current_speed) > 0:
            self.prev_angle = self.current_angle
            self.current_angle = mean_angle
        else:
            self.current_angle = self.prev_angle

        if self.show_vectors and  len(self.tracks) > 0:
            cv2.polylines(self.display, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
            #_draw_str(self.display, (20, self.height - 100), ' %d' % len(self.tracks), 2.0)

        if self.frame_idx % self.detect_interval == 0:
            mask = self.flow_mask.copy()
            for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                cv2.circle(mask, (x, y), 5, 0, -1)
            p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **self.settings['feature_params'])
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
        scale = 1.0 / self.reduction
        if not (self.gray_out_mask is None):
            self.display = cv2.bitwise_and(self.display, self.gray_out_mask)
        
        if not (self.current_angle is None):
            _draw_str(self.display, (self.width // 12, (15 * self.height) // 16),
                      'Direction: %d' % math.degrees(self.current_angle), scale)

        if not (self.current_distance is None):
            _draw_str(self.display, ((self.width) // 4, (15 * self.height) // 16),
                      'Traveled (pixels): %d' % self.current_distance, scale)
    
        if not (self.current_speed is None):
            _draw_str(self.display, ((2 * self.width) // 3, (15 * self.height) // 16),
                      'Current Speed (px / fr ): %d' % self.current_speed, scale)

    def draw_frame_info(self):
        scale = 1.0 / self.reduction
        cv2.putText(self.display, str(self.frame_idx), (self.width // 32, (15 * self.height) // 16),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    scale, (192, 192, 192), 2)


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


    def image_bounds_contains(self, point: np.ndarray) -> bool:
        point = (int(point[0]), int(point[1]))
        return cv2.pointPolygonTest(self.image_bounds, point, False) >= 0


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
        print(('User Supplied Data Directory does not exist or is Invalid ',
               ' Default Data Directory does not exist '))
        sys.exit(1)

    # use video_roi['hd'] for newer video
    runner = gpad_odometry(sys.argv[1], video_rois['hd'], data_path)
    loaded = runner.is_loaded()
    if not loaded:
        self.logger.info('Video Did not Load')
        exit(1)
    runner.run()

    cv2.destroyAllWindows()

#  cap.release()
#  cv2.waitKey(0)
#
