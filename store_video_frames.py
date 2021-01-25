import sys
import os
from pathlib import Path
from os.path import isfile, join
import itertools
import cv2




def store_video_frames(video_fqfn, parent_fqfn, dir_name = None, prefix = 'frame', format = '%04d', img_type = '.png'):
	if not Path(video_fqfn).exists() or (not Path(video_fqfn).is_file()):
		return None
	if not Path(parent_fqfn) or (not Path(parent_fqfn).is_dir()):
		return None
	if dir_name is None:
		dir_name = Path(video_fqfn).name
	cap = cv2.VideoCapture(video_fqfn)
	size = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
	run_vpd = True
	
	output_dir_fqfn = os.path.join(parent_fqfn, dir_name)
	os.makedirs(output_dir_fqfn,mode=511, exist_ok = True)
	
	fc = 0
	while run_vpd:
		run_vpd, captured_frame = cap.read()
		frame = cv2.rotate(captured_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
		file_name = prefix + format % (fc,) + img_type
		ofqfn = os.path.join(output_dir_fqfn, file_name)
		cv2.imwrite(ofqfn, frame)
		print(('frame', fc, ' of ', frame_count, ' Done'), end = '\n')
		fc +=1
		



if __name__ == '__main__':
	video_path = sys.argv[1]
	parent_fqfn = sys.argv[2]
	store_video_frames(video_path, parent_fqfn)
	