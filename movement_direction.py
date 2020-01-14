import sys
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import math
from wiitricity_settings import video_rois, initialize_settings_from_video_roi
from utils import drawString


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


class movement_direction:

    def __init__(self, video_source_info, video_roi):
        self.source_info = video_source_info
        self.source = None
        if isinstance(self.source_info, int) and self.source_info == 0:
            self.source = 'Camera'
        elif isinstance(self.source_info, str) and Path(self.source_info).exists() and Path(sys.argv[1]).is_file():
            self.source = self.source_info
        self._is_valid = not (self.source is None)
        self.row_range = (video_roi['row_low'], video_roi['row_high'])
        self.column_range = (video_roi['column_low'], video_roi['column_high'])
        self.settings = initialize_settings_from_video_roi(video_roi)
        self.width = video_roi['column_high'] - video_roi['column_low']
        self.height = video_roi['row_high'] - video_roi['row_low']

        self.feature_params = self.settings['feature_params']
        self.lk_params = self.settings['lk_params']
        self.max_dist = self.settings['max_distance']
        self.cap = None
        self._is_loaded = False
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

    def is_loaded(self):
        return self._is_loaded

    def is_valid(self):
        return self._is_valid

    def load(self):
        self.cap = cv.VideoCapture(self.source_info)
        self._is_loaded = True
        self.frame_count = 0
        self.prev_frame = None
        assert (self.width < int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH)))
        assert (self.height < int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)))

    def run(self):
        if not self._is_loaded: self.load()
        ret, captured_frame = self.cap.read()
        if not ret:
            return
        self.prev_frame = self.prepare_frame(captured_frame)
        print('frame: ', self.frame_count)
        self.frame_count = self.frame_count + 1

        while self.cap.isOpened():
            ret, captured_frame = self.cap.read()
            if not ret: break
            if self.debug: print('frame: ', self.frame_count)
            self.frame_count = self.frame_count + 1
            frame = self.prepare_frame(captured_frame)
            assert (not (self.prev_frame is None))
            self.update_features()
            self.get_flow(frame)
            self.prev_frame = frame.copy()

            disp_frame = self.draw_tracks(self.display, self.keypoints, self.new_points)
            cv.line(disp_frame, (0, self.height // 2), (self.width, self.height // 2), (255, 0, 0), 1)
            cv.line(disp_frame, (self.width // 2, 0), (self.width // 2, self.height), (255, 0, 0), 1)
            cv.imshow("Frame", disp_frame)

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
            if self.debug: print('max distance passed: reset ')
            avg_angle = self.measure_tracks(self.keypoints, self.new_points)
            self.direction.append(avg_angle)
            avg_angle = np.average(self.direction)
            print(('frame, direction:', self.frame_count, int(avg_angle)))


        # if few new points create features on the prev frame
        elif len(self.new_points) < self.min_features:
            self.get_features()
            if self.debug: print('copied old points to keypoints   ')
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
                if self.debug: print('too few features reset  ')
            else:
                # just copy new points to old points
                dim = np.shape(self.new_points)
                self.old_points = np.reshape(self.new_points, (-1, 1, 2))
                if self.debug: print('ok')

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

    def draw_tracks(self, frame, points1, points2):
        for i, (new, old) in enumerate(zip(points1, points2)):
            a, b = new.ravel()
            c, d = old.ravel()
            frame = cv.line(frame, (a, b), (c, d), (0, 255, 0), 1)

        if not (self.avg_angle is None):
            drawString(frame, self.avg_angle)

        return frame

    def measure_tracks(self, points1, points2):
        avg_angle = 0
        for i, (new, old) in enumerate(zip(points1, points2)):
            a, b = new.ravel()
            c, d = old.ravel()
            line = [c, d, a, b]
            angle = int(get_line_angle(line))
            avg_angle += angle
        avg_angle = avg_angle / len(points1)
        avg_angle = avg_angle % 180
        return avg_angle


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    runner = movement_direction(sys.argv[1], video_rois['one'])
    runner.load()
    loaded = runner.is_loaded()
    if not loaded:
        print('Video Did not Load')
        exit(1)
    runner.run()

#  video_file = sys.argv[1]
#  filename =
#
#  # Create some random colors
#  color = np.random.randint(0,255,(100,3))
#  # Take first frame and find corners in it
#  ret, old_frame = cap.read()
#  old_frame = old_frame[53:350, 6:708]
#  old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
#  p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
#  keypoints = np.reshape(p0, (-1, 2))
#  m0 = geometric_median_2d(keypoints)
#  m1 = m0
#
#  # Create a mask image for drawing purposes
#
#
#  fc = 1
#  directions = []
#  lhist = np.zeros((1,360), np.int)
#  while cap.isOpened():
#      ret, frame = cap.read()
#      frame = get_roi(frame, )
#      if not ret:
#          break
#
#      if prev_frame is None:
#          prev_frame = cv2.resize(frame, dsize=(w, h))
#          prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#          mask = np.zeros_like(old_frame)
#          continue
#
#      frame = cv2.resize(frame, dsize=(w, h))
#      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#  while(1):
#      ret,frame = cap.read()
#      if frame is None: break
#      frame = frame[53:350, 6: 708]
#
#      frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#      # calculate optical flow
#      p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
#      p0r, st, err = cv.calcOpticalFlowPyrLK(frame_gray, old_gray, p1, None, **lk_params)
#      d = abs(p0 - p0r).reshape(-1, 2).max(-1)
#      st = d < 1.0
#
#      if len(p1) < 1:
#          continue
#      if fc == 1:
#          cv.imwrite('/Users/arman/tmp/first.png', frame_gray)
#
#      # Select good points
#      good_new = p1[st==1]
#      good_old = p0[st==1]
#      # caclulate median flow position
#      keypoints = np.reshape(good_new, (-1, 2))
#      m1 = geometric_median_2d(keypoints)
#      path = [m1[0], m1[1], m0[0], m0[1]]
#      angle = int(get_line_angle(path))
#      dis = point_to_point_dist(m0, m1)
#      if dis < 1.0: continue
#      if dis > max_dist:continue
#          #old_points = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
#          #p0 = np.copy()
#      tcl = (0,0,255)
#      if dis > 10.0: tcl = (0,255,0)
#      mask = cv.line(mask, (int(m1[0]),int(m1[1])),(int(m0[0]),int(m0[1])), tcl, 2)
#
#      m0 = m1
#      # draw the tracks
#  #    print('------>' + str(fc))
#  #     for i,(new,old) in enumerate(zip(good_new,good_old)):
#  #         a,b = new.ravel()
#  #         c,d = old.ravel()
#  #         line = [a,b,c,d]
#    #  #         angle = int(get_line_angle(line))
#    #  #         dis = point_to_point_dist((a,b),(c,d)) // 1
#  #         lhist[0][angle] = lhist[0][angle] + dis
#  #         if dis > 0:
#  #          #   print (str(angle) + '......' + str(dis))
#  #             directions.append(angle)
#  #         cl = bgrFromHue(angle)
#  #         mask = cv.line(mask, (a,b),(c,d), cl, 2)
#  #         frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
# #         cv.circle(frame, m0, 10, (0, 0, 255), -1)
#  #    print('<------' + str(fc))
#      img = cv.add(frame,mask)
#      cv.namedWindow('frame', cv.WINDOW_NORMAL)
#      cv.imshow('frame',img)
#      fc = fc + 1
#      k = cv.waitKey(30) & 0xff
#      if k == 27:
#          break
#      # Now update the previous frame and previous points
#      old_gray = frame_gray.copy()
#      p0 = good_new.reshape(-1,1,2)
#
#  _ = plt.hist(lhist , fc='k', ec='k')  # arguments are passed to np.histogram
#  plt.title(filename)
#  plt.show()
#  cv.waitKey(0)
#  cv.destroyAllWindows()
#  cap.release()
