# This code uses the pygrabber library to grab successive frames from a connected camera, reverses the color scheme and shows live video on the screen
"""
Created on Tue Nov 12 14:00:32 2019

@author: shrenik.vora
"""
from pygrabber.dshow_graph import FilterGraph
import cv2
import threading
import numpy as np

image_done = threading.Event()
image_grabbed = None


def img_change(image):
    global image_done
    global image_grabbed
    image_grabbed = np.flip(image, 2)
    image_done.set()


graph = FilterGraph()
# List available cameras
cam_list = graph.get_input_devices()
counter = 0
for i in cam_list:
    print(counter, i)
    counter += 1
camera_idx = input('Enter the camera index number ')
graph.add_video_input_device(int(camera_idx))

# Show videostream
graph.add_sample_grabber(img_change)
graph.add_null_render()
graph.prepare_preview_graph()
graph.run()
print('Press q to stop')
while True:
    graph.grab_frame()
    image_done.wait(1)
    cv2.imshow('Cam Stream', image_grabbed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

graph.stop()
cv2.destroyAllWindows()
