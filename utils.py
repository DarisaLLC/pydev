import cv2
import numpy as np


def circleContainsPoint(center, radius_2, point):
    dx = center[0] - point[0]
    dy = center[1] - point[1]
    dd = dy * dy + dx * dx
    return dd < radius_2


def circleContainsBoundingRect(center, radius_2, bb):
    pts = []
    pts.append((bb[0], bb[1]))
    pts.append((bb[0] + bb[2], bb[1]))
    pts.append((bb[0] + bb[2], bb[1] + bb[3]))
    pts.append((bb[0], bb[1] + bb[3]))
    for pt in pts:
        if not circleContainsBoundingRect(center, radius_2, pt): return False
    return True
