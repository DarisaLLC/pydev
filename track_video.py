import numpy as np
import cv2
import os
import time
from imutils.video import FPS
import math
import random


def get_speed(mat):
    vx = mat[0, 0]
    vy = mat[1, 0]
    vz = mat[2, 0]
    speed = vx**2+vy**2+vz**2
    speed = math.sqrt(speed)
    return speed


def get_rotation(mat, axis=0):
    if axis == 0:
        v0 = np.array([[1.], [0.], [0.]], dtype=np.float32)
    elif axis == 1:
        v0 = np.array([[0.], [1.], [0.]], dtype=np.float32)
    else:
        v0 = np.array([[0.], [0.], [1.]], dtype=np.float32)

    v1 = np.dot(mat, v0)
    dot = np.vdot(v0, v1)
    a = np.arccos(dot)
    return a


def draw_tracks(canvas, points1, points2):
    for i, (new, old) in enumerate(zip(points1, points2)):
        a, b = new.ravel()
        c, d = old.ravel()
        canvas = cv2.line(canvas, (a, b), (c, d), (0, 255, 0), 1)
    return canvas


def get_mean_distance_2d(features1, features2):
    num_features = min((len(features1), len(features2)))
    features1 = np.reshape(features1, (num_features, 2))
    features2 = np.reshape(features2, (num_features, 2))

    features = zip(features1, features2)
    n = 0
    dist = 0
    for f1, f2 in features:
        dx = f1[0]-f2[0]
        dy = f1[1]-f2[1]
        d = math.sqrt(dx**2+dy**2)
        dist += d
        n += 1

    if n == 0:
        return 0

    dist /= n
    return dist


def update_motion(points1, points2, Rp, Tp, scale=1.0):
    p1 = np.reshape(np.array(points1, dtype=np.float32), (-1, 1, 2))
    p2 = np.reshape(np.array(points2, dtype=np.float32), (-1, 1, 2))
    E, mask = cv2.findEssentialMat(p1, p2,
                                   cameraMatrix=cam_mat, method=cv2.RANSAC, prob=0.999, threshold=1.0, mask=None)
    points, R, T, newmask = cv2.recoverPose(E, p1, p2, cameraMatrix=cam_mat, mask=mask)

    Tp = Tp + np.dot(R, T)*scale
    Rp = np.dot(R, Rp)
    return Rp, Tp


def draw_path(cnv, points, R, rotate=True, drawVector=True, scale=0.1):
    dim = np.shape(cnv)
    cnv_h = dim[0]
    cnv_w = dim[1]

    #clear canvas
    cnv = np.zeros(dim, dtype=np.uint8)

    pos_final = points[-1]
    x_final = pos_final[0, 0]
    y_final = pos_final[1, 0]

    for i, p2 in enumerate(points):
        if i == 0:
            continue
        p1 = points[i-1]
        x1 = int((p1[0, 0] - x_final) * scale + cnv_w / 2)
        y1 = int((p1[1, 0] - y_final) * scale + cnv_h / 2)
        x2 = int((p2[0, 0] - x_final) * scale + cnv_w / 2)
        y2 = int((p2[1, 0] - y_final) * scale + cnv_h / 2)
        cv2.line(cnv, (x1, y1), (x2, y2), (0, 255, 0), 1)

    if drawVector is True:
        vec = np.transpose(np.array([0., -1., 0.], dtype=np.float32))
        vec = np.dot(R, vec)
        print(vec)
        cv2.line(cnv, (cnv_w//2, cnv_h//2),
                 (int(16*vec[0]+cnv_w/2), int(16*vec[1]+cnv_h/2)),
                 (255, 255, 255), 1)

    return cnv


min_features = 300
max_dist = 25

fps = FPS().start()

# image dimensions
h = 1080//2
w = 1920//2

canvas = np.zeros((h, w, 3), dtype=np.uint8)

# calibrated camera parameters
cam_mat = np.array([[949.38174439, 0., 479.90568209],
                    [0., 953.41397342, 261.82744181],
                    [0., 0., 1.]], dtype=np.float32)
dist_coeff = np.array((2.78460254e-01, -2.47492728e+00, -4.05126422e-04, 3.71587977e-03, 6.17755379e+00),
                      dtype=np.float32)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=8000,
                      qualityLevel=0.01,
                      minDistance=5,
                      blockSize=3,
                      useHarrisDetector=False,
                      k=0.04)

Rpos = np.eye(3, 3, dtype=np.float32)
Tpos = np.zeros((3, 1), dtype=np.float32)

cap = cv2.VideoCapture("/Users/arman/PycharmProjects/pydev/projects/wiic/video/2019_11_11/fiducial_curved_approach_2.mp4")
prev_frame = None
prev_gray = None

keypoint_dist = 0

old_points = []
new_points = []

keypoints = []

path = []
fc = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    print(('frame: ',fc))
    fc = fc + 1
    if prev_frame is None:
        prev_frame = cv2.resize(frame, dsize=(w, h))
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        continue

    frame = cv2.resize(frame, dsize=(w, h))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if keypoint_dist > max_dist:
        # update motion
        if len(keypoints) > 7 and len(new_points) > 7:
            old_pos = np.copy(Tpos)  # copy old position for drawing of path
            Rpos, Tpos = update_motion(new_points, keypoints, Rpos, Tpos, scale=keypoint_dist)

            # draw path
            path.append(Tpos)
            canvas = draw_path(canvas, path, Rpos)

        # select new keypoints
        keypoint_dist = 0
        old_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        keypoints = np.copy(old_points)
        print('max distance passed: reset ')
    elif len(new_points) < min_features:
        # update motion
        if len(keypoints) > 7 and len(new_points) > 7:
            old_pos = np.copy(Tpos)  # copy old position for drawing of path
            Rpos, Tpos = update_motion(new_points, keypoints, Rpos, Tpos, scale=keypoint_dist)

            # draw path
            path.append(Tpos)
            canvas = draw_path(canvas, path, Rpos)

        # select new keypoints
        keypoint_dist = 0
        old_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        keypoints = np.copy(old_points)
        print('copied old_points to keypoints ')
    else:
        # check number of features in each quadrant to ensure a good distribution of features across entire image
        nw = 0
        ne = 0
        sw = 0
        se = 0
        for x, y in new_points:
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

        num_features = min((nw, ne, sw, se))
        if num_features < min_features // 4:
            # update motion
            if len(keypoints) > 7 and len(new_points) > 7:
                old_pos = np.copy(Tpos)  # copy old position for drawing of path
                Rpos, Tpos = update_motion(new_points, keypoints, Rpos, Tpos, scale=keypoint_dist)

                # draw path
                path.append(Tpos)
                canvas = draw_path(canvas, path, Rpos)

            # select new keypoints
            keypoint_dist = 0
            old_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
            keypoints = np.copy(old_points)
            print('too few features reset  ')
        else:
            dim = np.shape(new_points)
            old_points = np.reshape(new_points, (-1, 1, 2))
            print('ok')


    new_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, old_points, None, **lk_params)
    keypoints = np.reshape(keypoints, (-1, 1, 2))  # TODO find out why this is necessary?!
    old_points = old_points[status == 1]
    new_points = new_points[status == 1]
    keypoints = keypoints[status == 1]

    keypoint_dist += get_mean_distance_2d(old_points, new_points)

    frame = draw_tracks(frame, keypoints, new_points)
    cv2.line(frame, (0, h // 2), (w, h // 2), (255, 0, 0), 1)
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 0, 0), 1)
    cv2.imshow("Frame", frame)
    cv2.imshow("Path", canvas)

    prev_gray = gray.copy()
    key = cv2.waitKey(1) & 0xFF

    if key == ord("a"):
        print("Finding new features..")
        points1 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)

    if key == ord("r"):
        print("Resetting path...")
        Rpos = np.eye(3, 3, dtype=np.float32)
        Tpos = np.zeros((3, 1), dtype=np.float32)
        path = np.zeros((h, w, 3), dtype=np.uint8)

    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
cap.release()