import cv2
import numpy as np



def make_Hough_transform(img_path, kernel_size=3, sigmaX=0, low_threshold=50, high_threshold=150, rho=1,
						 min_votes=15, min_line_length=10, max_line_gap=15, num_of_votes=15, filter_norails_on=False,
						 show_orig_image=True, plot_only_rales=False):
	"""

	sigmaX = Gaussian Kernel Standard Deviation
	kernel_size = Gaussian Kernel size (for Gaussian blur)
	low_threshold = first threshold for the hysteresis procedure (for Canny detector)
	high_threshold = second threshold for the hysteresis procedure (for Canny detector)
	rho = distance resolution in pixels of the Hough grid
	num_of_votes (= threshold) = minimum number of votes (intersections in Hough grid cell)
	"""


	image = cv2.imread(img_path)
	theta = np.pi / 180  # angular resolution in radians of the Hough grid
	line_image = np.copy(image) * 0  # creating a blank to draw lines on
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigmaX)
	edges = cv2.Canny(gray, low_threshold, high_threshold)

	# Run Hough on edge detected image
	# Output "lines" is an array containing endpoints of detected line segments
	lines = cv2.HoughLinesP(edges, rho, theta, num_of_votes, np.array([]),
	                        min_line_length, max_line_gap)
	for line in lines:
	    for x1,y1,x2,y2 in line:
	    	cv2.line(line_image,(x1,y1),(x2,y2),(0,128,0), 5)

	# Draw the lines on the image
	lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)

	if show_orig_image:
		cv2.imshow('Edges', edges)
		cv2.waitKey(0)
		cv2.imshow('Lines', lines_edges)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	else:
		print('Hough transform completed.')
		print('{} line(s) were found\n'.format(len(lines)))

	if filter_norails_on:
		rails = filter_norails(image, lines)
		print('{} rail(s) remained only.\n'.format(len(rails)))
		if plot_only_rales:
			line_image = np.copy(image) * 0  # creating a blank to draw lines on
			for rail in rails:
				for x1,y1,x2,y2 in rail:
					cv2.line(line_image,(x1,y1),(x2,y2),(0,128,0),5)
			# Draw the lines on the image
			lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
			cv2.imshow('Norails', lines_edges)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		else:
			print('Filtration was completed.')
		return rails
	return lines



def filter_norails(image, lines, permissable_error=10):

	assert lines.shape[0] > 2, 'Rails are not found'
	
	low_permission = image.shape[0] - permissable_error
	high_permission = image.shape[0] + permissable_error
	line_num = 0
	rails_coords = []
	for line in lines:
		for coords in line:
			reduced = list(filter(lambda c: c >= low_permission and c <= high_permission, coords))
			if reduced != []:
				rails_coords.append(lines[line_num])
			line_num += 1
	return rails_coords


if __name__ == '__main__':
	img_path = '/Volumes/medvedev/_SP/2019_09_06_BMWi3_natika_camera_images/empty_pad/empty_pad3/image00381.png'
	lines = make_Hough_transform(img_path)