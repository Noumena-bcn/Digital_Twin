"""
Functions:
-four_point_transform_geo
-homography
-order_points (limited)
-sort_markers (general)

Required:
-OpenCV
-numpy
"""

import numpy as np
import cv2

def four_point_transform_geo(corners_image, corners_loc):
	"""
	Inputs:
	corners_loc: ndarray
	pts: ndarray
	Output:
	Homography np matrix
	"""
	pts = np.array(corners_image, dtype="float32")
	dst = np.array(corners_loc, dtype="float32")
	(tl, tr, br, bl) = pts

	H, status = cv2.findHomography(pts, dst)

	return H


def homography(pts_img, H):
	"""
	Inputs: 
		pts_img: list of tuples (x,y) or ndarray
		H: Homography matrix from cv2.findHomography
	Output:
		2 dimensional list of mapped coordinates
	"""
	points = np.asarray(pts_img, dtype=np.float32)
	points = np.array([points])
	try:
		pts_res = cv2.perspectiveTransform(points, H)
		pointsOut = np.asarray(pts_res)
		pointsOut = pointsOut.tolist()[0]
		return pointsOut
	except Exception as e:
		print(e)
		if len(points.shape) < 3:
			print("Error occured because the input vector doesn't have the required shape, it must be 2 dimensional.")

def order_points(pts):
	"""
	takes a list of tuples (4,2) or ndarray (4,2)
	returns an ndarray (4, 2) of coordinates that will be ordered:
	1: top-left,
	2: top-right, 
	3: bottom-right, 
	4: bottom-left
	works only if bottom_left is to the left of top_right
	"""
	pts = np.array(pts)
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	print(s)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# the top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	print(diff)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def sort_markers(markers):
	"""
	takes a list of tuples (4,2) or ndarray (4,2)
	returns an ndarray (4, 2) of coordinates that will be ordered:
	1: top-left,
	2: top-right, 
	3: bottom-right, 
	4: bottom-left
	"""
	points = np.array(markers)
	center = points.mean(axis=0)
	moved = points - center
	r = np.linalg.norm(moved, axis=1)
	# print (r)
	y = moved[:, 1]
	print(y)
	x = moved[:, 0]
	print(x)
	arccos = np.arccos(x/r)
	sign = np.where(y >= 0, 1, -1)
	theta = arccos * sign
	print(theta)
	key = theta.argsort()
	ordered = np.zeros((len(key), 2), dtype=int)
	for i in range(len(key)):
		ordered[i] = points[key[i]]
	return ordered