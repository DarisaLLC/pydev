import math
import numpy as np
import os
from datetime import datetime
import dlib
import cv2
import time
import argparse


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



def processing_info(np_arr, name=None, elapsed=None):
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
