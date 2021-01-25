import math
import numpy as np
import os
from datetime import datetime
import dlib
import cv2
import time
import argparse

'''
Create a mask like this.
diagonal ratio is  ration of ... to ...-----...

 ...------...
   /      \
  /        \
  ----------

'''


def region_of_interest(img, diagonal_ratio = 2.5):
	rows, cols, channels = img.shape
	mask = np.zeros((rows, cols), dtype = np.uint8)
	left = int(cols / diagonal_ratio)
	right = int(cols - left)
	
	poly = np.array([[
		(left, 50), (right, 50), (cols - 1, rows - 1), (0, rows - 1)]], np.int32)
	
	cv2.fillConvexPoly(mask, poly, 255)
	return mask


'''
Create a mask like this.
Higher portion

  ----------
  ----------

'''


def vertical(img, top_portion = 0.667):
	rows, cols, channels = img.shape
	mask = np.zeros((rows, cols), dtype = np.uint8)
	top = int(rows * top_portion)
	
	poly = np.array([[
		(0, top), (cols - 1, top), (cols - 1, rows - 1), (0, rows - 1)]], np.int32)
	
	cv2.fillConvexPoly(mask, poly, 255)
	return mask


# does the match, if it's good returns the homography transform
def find(des, kp, img_des, img_kp,MIN_MATCH_COUNT = 50):
	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
	
	# Match descriptors.
	matches = bf.match(des, img_des)
	
	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x: x.distance)
	
	print("matches are found - %d/%d" % (len(matches), MIN_MATCH_COUNT))
	
	if len(matches) > MIN_MATCH_COUNT:
		src_pts = np.float32([kp[m.queryIdx].pt for m in matches[:MIN_MATCH_COUNT]]).reshape(-1, 1, 2)
		dst_pts = np.float32([img_kp[m.trainIdx].pt for m in matches[:MIN_MATCH_COUNT]]).reshape(-1, 1, 2)
		
		# get the transformation between the flat fiducial and the found fiducial in the photo
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		matchesMask = mask.ravel().tolist()
		# return the transform
		return M, matchesMask
	else:
		print("Not enough matches are found - %d/%d" % (len(matches), MIN_MATCH_COUNT))
		return None, None


# draws a box round the fiducial
def draw_outline(display, M, h, w):
	# array containing co-ords of the fiducial
	pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
	# transform the coords of the fiducial onto the picture
	dst = cv2.perspectiveTransform(pts, M)
	# draw a box around the fiducial
	cv2.polylines(display, [np.int32(dst)], True, (255, 0, 0), 5, cv2.LINE_AA)


def get_rotatedRect_angle(rr):
	return np.radians(90.0 - rr[2]) if rr[1][0] < rr[1][1] else np.radians(-rr[2])


def _draw_str(dst, target, s, scale):
	x, y = target
	cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_COMPLEX_SMALL, scale, (255, 255, 255), thickness = 2,
				lineType = cv2.LINE_AA)
	cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, scale, (64, 64, 64), lineType = cv2.LINE_AA)


class Time:
	"""
	Class for displaying elapsed time.
	"""
	
	def __init__(self):
		self.start = datetime.now()
	
	def elapsed_display(self):
		time_elapsed = self.elapsed()
		print("Time elapsed: " + str(time_elapsed))
	
	def elapsed(self):
		self.end = datetime.now()
		time_elapsed = self.end - self.start
		return time_elapsed


def processing_info(np_arr, name = None, elapsed = None):
	"""
	Display information (shape, type, max, min, etc) about a NumPy array.
	Args:
	  np_arr: The NumPy array.
	  name: The (optional) name of the array.
	  elapsed: The (optional) time elapsed to perform a filtering operation.
	"""
	
	if name is None:
		name = "NumPy Array"
	if elapsed is None:
		elapsed = "---"
	
	print("%-20s | Time: %-14s  Type: %-7s Shape: %s" % (name, str(elapsed), np_arr.dtype, np_arr.shape))


def roiPts(pts):
	s = np.sum(pts, axis = 1)
	(x, y) = pts[np.argmin(s)]
	(xb, yb) = pts[np.argmax(s)]
	return [(x, y), (xb, yb)]


def point_2d_norm(point_0):
	"""Returns the norm of a 2d point.
	:param point_0:
	:return:
	"""
	return (point_0[0] ** 2 + point_0[1] ** 2) ** 0.5


def point_is_near_point(point_0, point_1, error = 0.001):
	"""Return true iff the two points are near each other.
	:param point_0:
	:param point_1:
	:param error:
	:return:
	"""
	return point_2d_norm(point_0 - point_1) < error


def point_line_projection(point, line_start, line_end):
	"""Return the projection of the point on the line.
	:param point:
	:param line_start:
	:param line_end:
	:return:
	"""
	# Determine the projection of the point to the line
	v = line_end - line_start
	w = point - line_start
	return line_start + np.dot(v, w) * v / (point_2d_norm(v) ** 2)


def point_line_distance_and_projection(point, line_start, line_end):
	"""Return the distance from the point to the projection of the point on the line.
	:param point:
	:param line_start:
	:param line_end:
	:return:
	"""
	# Get the projection of the point
	projection = point_line_projection(point, line_start, line_end)
	# Return the distance between the projection and the point
	return point_2d_norm(projection - point), projection


def point_is_on_line(point, line_start, line_end, error = 0.001):
	"""Return true iff the point intersects the line.
	:param point:
	:param line_start:
	:param line_end:
	:param error:
	:return:
	"""
	# We take the wedge- and dot product to determine if the point intersects the line
	v = line_end - point
	w = point - line_start
	return abs(np.linalg.det(np.column_stack((v, w)))) < error and np.dot(v, w) >= 0


def point_projection_is_on_line(point, line_start, line_end, error = 0.001):
	"""Return true iff the point projection intersects the line.
	:param point:
	:param line_start:
	:param line_end:
	:param error:
	:return:
	"""
	# Determine the projection of the point to the line
	return point_is_on_line(point_line_projection(point, line_start, line_end), line_start, line_end, error)


def is_collinear(point_0, point_1, point_2, error = 0.001):
	"""Return true iff the three points are collinear.
	:param point_0:
	:param point_1:
	:param point_2:
	:param error:
	:return:
	"""
	# We take the wedge product to determine if the three points are collinear
	v = point_2 - point_1
	w = point_0 - point_1
	return np.linalg.det(np.column_stack((v, w))) < error


def line_to_unit_interval(point, line_start, line_end, error = 0.001):
	"""Put a point on the unit interval of a line.
	:param point:
	:param line_start:
	:param line_end:
	:param error:
	:return:
	"""
	if point_is_on_line(point, line_start, line_end, error):
		return point_2d_norm(point - line_start) / point_2d_norm(line_end - line_start)
	else:
		return None


def line_embedding(unit, line_start, line_end):
	"""Embed a unit on to the line.
	:param unit:
	:param line_start:
	:param line_end:
	:return:
	"""
	if 0.0 <= unit <= 1:
		return (1 - unit) * line_start + unit * line_end
	else:
		return None


def bgrFromHue(degrees):
	hsv = np.zeros((1, 1, 3), np.uint8)
	hsv[0, 0, 0] = ((degrees % 180) * 256) / 180.0
	hsv[0, 0, 1] = 255
	hsv[0, 0, 2] = 255
	bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
	tp = tuple([int(x) for x in bgr[0, 0, :]])
	return tp


def get_line_angle(line):
	x1, y1, x2, y2 = np.array(line, dtype = np.float64)
	radians = np.arctan2(y2 - y1, x2 - x1)
	return radians


def circular_mean(weights, angles):
	x = y = 0.
	for angle, weight in zip(angles, weights):
		x += math.cos(math.radians(angle)) * weight
		y += math.sin(math.radians(angle)) * weight
	
	mean = math.degrees(math.atan2(y, x))
	return mean


'''
Choices are lsd or hough. hough appeared to work better
'''


def compute_lines(image, expected_orientation, length_limit, vertical_horizon, dlogger, dsettings):
	"""

	sigmaX = Gaussian Kernel Standard Deviation
	kernel_size = Gaussian Kernel size (for Gaussian blur)
	low_threshold = first threshold for the hysteresis procedure (for Canny detector)
	high_threshold = second threshold for the hysteresis procedure (for Canny detector)
	rho = distance resolution in pixels of the Hough grid
	num_of_votes (= threshold) = minimum number of votes (intersections in Hough grid cell)
	"""
	sigmaX = 1.0
	kernel_size = 3
	low_threshold = 30
	high_threshold = 150
	rho = 1
	min_votes = 15
	min_line_length = 10
	max_line_gap = 15
	theta = np.pi / 360  # angular resolution in radians of the Hough grid
	
	blur_gray = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX)
	edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
	
	# Run Hough on edge detected image
	# Output "lines" is an array containing endpoints of detected line segments
	lines = cv2.HoughLinesP(edges, rho, theta, min_votes, np.array([]),
							min_line_length, max_line_gap)
	
	# Remove singleton dimension
	lines = lines[:, 0]
	
	return filter_lines(lines, expected_orientation, length_limit, vertical_horizon, dlogger, dsettings)


def merge_lines(lines, max_merging_angle = np.pi * 180 / 5, max_endpoint_distance = 1):
	merged = []
	merged_segments = []
	for i, segment_i in enumerate(lines):
		if i in merged:
			continue
		collinears = [i]
		for j in range(i + 1, len(lines)):
			segment_j = lines[j]
			if utils.segments_collinear(segment_i, segment_j, max_angle = max_merging_angle,
										max_endpoint_distance = max_endpoint_distance):
				collinears.append(j)
		
		merged_segment = utils.merge_segments(lines[collinears])
		merged_segment = [int(m) for m in merged_segment]
		merged_segments.append(merged_segment)
		
		for index in collinears:
			if index not in merged:
				merged.append(index)
	
	return merged, np.array(merged_segments)


'''
Line Filtering
mask region is a rectangle in tl image system
represented by tl and br coords

'''


def filter_lines(lines, expected_orientation, expected_size, vertical_horizon, dlogger, dsettings):
	N = len(lines)
	dlogger.info('Initial Line Count: ' + str(N))
	dx = lines[:, 2] - lines[:, 0]
	dy = lines[:, 3] - lines[:, 1]
	xc = (lines[:, 2] + lines[:, 0]) / 2.0
	yc = (lines[:, 3] + lines[:, 1]) / 2.0
	
	# setup region for line processing
	mask = yc > vertical_horizon
	lines = lines[mask]
	dlogger.info('Post Horizon filter Line Count: ' + str(len(lines)))
	
	dx = lines[:, 2] - lines[:, 0]
	dy = lines[:, 3] - lines[:, 1]
	xc = (lines[:, 2] + lines[:, 0]) / 2.0
	yc = (lines[:, 3] + lines[:, 1]) / 2.0
	
	directions = np.mod((np.arctan2(dy, dx) + np.pi), np.pi)
	lengths = np.sqrt(dx * dx + dy * dy)
	if dsettings['restrict_to_view_angle']:
		mask = np.mod(directions, expected_orientation) < 0.01
		lines = lines[mask]
		dx = dx[mask]
		dy = dy[mask]
		xc = xc[mask]
		yc = yc[mask]
		directions = directions[mask]
		lengths = lengths[mask]
		dlogger.info('Post Movement Direction filter Line Count: ' + str(len(lines)))
	
	N = len(lines)
	#    out = np.zeros((N,N), dtype='float')
	iterator = itertools.combinations(range(N), 2)
	rects = []
	cands = []
	
	# Check on size
	# Check on parallelism cosine distance -> 1
	# Check on overlapness
	# Check on distance between centers
	# Check on whether mid to mid is prependicular to either line
	rng = np.random.RandomState(0)
	X = rng.random_sample((2, 2))
	angle_thr = np.pi * 180 / 5
	
	for i, j in iterator:
		
		## 3 lines: i, j and mid_i to mid_j
		## first check if i and j are close to parallel
		# form mid_i_2_mid_j
		# check if mid_i_2_mid_j and either i or j are prependicular to each other
		ia = get_line_angle(lines[i])
		ja = get_line_angle(lines[j])
		if math.fabs(ia - ja) > angle_thr:
			continue
		m2m = (xc[i], yc[i], xc[j], yc[j])
		ma = get_line_angle(m2m)
		pp = ma - ia
		if pp < 0:
			pp = pp + np.pi
		if math.fabs(pp - np.pi / 2) > angle_thr:
			continue
		
		wide, short = i, j
		if lengths[wide] < lengths[short]:
			wide, short = j, i
		cands.append((wide, short))
		
		(xx1, yy1, xx2, yy2) = lines[wide]
		p1 = np.array((xx1, yy1))
		p2 = np.array((xx2, yy2))
		(x1, y1, x2, y2) = lines[short]
		p3 = np.array((x1, y1))
		p4 = np.array((x2, y2))
		d1, p13 = point_line_distance_and_projection(p3, p1, p2)
		d2, p14 = point_line_distance_and_projection(p4, p1, p2)
		
		#        not_overlaps_in_x = np.maximum(p13[0], p14[0]) > np.minimum(p1[0],p2[0]) or np.maximum(p1[0], p2[0]) > np.minimum(p13[0],p14[0])
		#        not_overlaps_in_y = np.maximum(p13[1], p14[1]) > np.minimum(p1[1],p2[1]) or np.maximum(p1[1], p2[1]) > np.minimum(p13[1],p14[1])
		#        if not_overlaps_in_x or not_overlaps_in_y: continue
		
		points = [p1, p2, p3, p4]
		points = np.asarray(points, dtype = np.float32)
		rr = cv2.minAreaRect(points)
		(x, y), (width, height), angle = rr
		aspect_ratio = min(width, height) / max(width, height)
		if aspect_ratio < 0.33:
			continue
		if angle < 0:
			angle = angle + 360.0
		slant = angle % (expected_orientation * 180 / np.pi)
		if slant > 5:
			continue
		
		rects.append(rr)
	
	return (rects, lines, directions, xc, yc, None)


def angle_diff(angle1: float, angle2: float) -> float:
	"""Computes the angle difference between the input arguments.

	.. note:: The resulting angle difference is in [0, pi * 0.5]

	:param angle1: First angle expressed in radiants.
	:param angle2: Second angle expressed in radiants.
	:return: Angle difference between the input parameters. This angle represents the smallest positive angle between the input parameters.
	"""
	d_angle = np.abs(angle1 - angle2)
	d_angle = d_angle % np.pi
	if d_angle > np.pi * 0.5:
		d_angle -= np.pi
	return np.abs(d_angle)


def get_mid_point(line):
	return np.array([(line[0] + line[2]) / 2, (line[1] + line[3]) / 2])


def get_distance(pt1, pt2):
	diff = pt1 - pt2


from collections.abc import Hashable


def uniques(iterable):
	seen_hashable = set()
	seen_unhashable = []
	for item in iterable:
		if isinstance(item, Hashable):
			if item not in seen_hashable:
				yield item
				seen_hashable.add(item)
		else:
			if item not in seen_unhashable:
				yield item
				seen_unhashable.append(item)
