import math
import numpy as np
import os
from datetime import datetime
import skimage.future as sk_future
import skimage.color as sk_color
import skimage.morphology as sk_morphology
import skimage.segmentation as sk_segmentation


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
  
def filter_kmeans_segmentation(np_img, compactness=10, n_segments=16):
  """
  Use K-means segmentation (color/space proximity) to segment RGB image where each segment is
  colored based on the average color for that segment.
  Args:
    np_img: Binary image as a NumPy array.
    compactness: Color proximity versus space proximity factor.
    n_segments: The number of segments.
  Returns:
    NumPy array (uint8) representing 3-channel RGB image where each segment has been colored based on the average
    color for that segment.
  """
  t = Time()
  labels = sk_segmentation.slic(np_img, compactness=compactness, n_segments=n_segments)
  result = sk_color.label2rgb(labels, np_img, kind='avg')
  processing_info(result, "K-Means Segmentation", t.elapsed())
  return result

def filter_rag_threshold(np_img, compactness=3, n_segments=8, threshold=9):
  """
  Use K-means segmentation to segment RGB image, build region adjacency graph based on the segments, combine
  similar regions based on threshold value, and then output these resulting region segments.
  Args:
    np_img: Binary image as a NumPy array.
    compactness: Color proximity versus space proximity factor.
    n_segments: The number of segments.
    threshold: Threshold value for combining regions.
  Returns:
    NumPy array (uint8) representing 3-channel RGB image where each segment has been colored based on the average
    color for that segment (and similar segments have been combined).
  """
  t = Time()
  labels = sk_segmentation.slic(np_img, compactness=compactness, n_segments=n_segments)
  g = sk_future.graph.rag_mean_color(np_img, labels)
  labels2 = sk_future.graph.cut_threshold(labels, g, threshold)
  result = sk_color.label2rgb(labels2, np_img, kind='avg')
  processing_info(result, "RAG Threshold", t.elapsed())
  return result


def filter_hysteresis_threshold(np_img, low = 50, high = 100, output_type = "uint8"):
	"""
	Apply two-level (hysteresis) threshold to an image as a NumPy array, returning a binary image.
	Args:
	  np_img: Image as a NumPy array.
	  low: Low threshold.
	  high: High threshold.
	  output_type: Type of array to return (bool, float, or uint8).
	Returns:
	  NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above hysteresis threshold.
	"""
	
	hyst = sk_filters.apply_hysteresis_threshold(np_img, low, high)
	if output_type == "bool":
		pass
	elif output_type == "float":
		hyst = hyst.astype(float)
	else:
		hyst = (255 * hyst).astype("uint8")
	
	return hyst


def filter_threshold(np_img, threshold, output_type = "uint8"):
	"""
	Return mask where a pixel has a value if it exceeds the threshold value.
	Args:
	  np_img: Binary image as a NumPy array.
	  threshold: The threshold value to exceed.
	  output_type: Type of array to return (bool, float, or uint8).
	Returns:
	  NumPy array representing a mask where a pixel has a value (T, 1.0, or 255) if the corresponding input array
	  pixel exceeds the threshold value.
	"""
	t = Time()
	result = (np_img > threshold)
	if output_type == "bool":
		pass
	elif output_type == "float":
		result = result.astype(float)
	else:
		result = result.astype("uint8") * 255
	processing_info(result, "Threshold", t.elapsed())
	return result


def filter_grays(rgb, tolerance = 15, output_type = "uint8"):
	"""
	Create a mask to filter out pixels where the red, green, and blue channel values are similar.
	Args:
	  np_img: RGB image as a NumPy array.
	  tolerance: Tolerance value to determine how similar the values must be in order to be filtered out
	  output_type: Type of array to return (bool, float, or uint8).
	Returns:
	  NumPy array representing a mask where pixels with similar red, green, and blue values have been masked out.
	"""
	t = Time()
	(h, w, c) = rgb.shape
	
	rgb = rgb.astype(np.int)
	rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
	rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
	gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
	result = ~(rg_diff & rb_diff & gb_diff)
	
	if output_type == "bool":
		pass
	elif output_type == "float":
		result = result.astype(float)
	else:
		result = result.astype("uint8") * 255
	processing_info(result, "Filter Grays", t.elapsed())
	return result

