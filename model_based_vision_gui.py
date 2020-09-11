from gui import *
import cv2
import numpy as np

sigma = 0.33
threshold = 1
min_line_length = 100
max_line_gap = 10

def process():
    global image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    v = np.median(gray)
    lower_thresh = int(max(0, (1.0-sigma)*v))
    upper_thresh = int(min(255, (1.0+sigma)*v))
    edges = cv2.Canny(gray, lower_thresh, upper_thresh)
    result = cv2.HoughLinesP(
        edges, 5, 5*np.pi/180, threshold, min_line_length, max_line_gap)
    if result is not None:
        for l in result:
            if len(l)!=1: print l
    if result is None: lines = []
    else: lines = [list(l[0]) for l in result]
    if show_edges():
        image_to_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    else: image_to_display = image.copy()
    if show_lines():
        for x1, y1, x2, y2 in lines:
            cv2.line(image_to_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
    get_window().show_image(image_to_display)

def capture_command():
    global image
    message("")
    camera = cv2.VideoCapture(0)
    return_value, image = camera.read()
    camera.release()
    process()

def dummy_command():
    message("")

def decrement_threshold():
    global threshold
    threshold -= 10
    print threshold
    process()

def increment_threshold():
    global threshold
    threshold += 10
    print threshold
    process()

def decrement_min_line_length():
    global min_line_length
    min_line_length -= 10
    print min_line_length
    process()

def increment_min_line_length():
    global min_line_length
    min_line_length += 10
    print min_line_length
    process()

def decrement_max_line_gap():
    global max_line_gap
    max_line_gap -= 10
    print max_line_gap
    process()

def increment_max_line_gap():
    global max_line_gap
    max_line_gap += 10
    print max_line_gap
    process()

add_button(0, 0, "Capture", capture_command, process)
show_edges = add_checkbox(0, 1, "Edges?", process)
show_lines = add_checkbox(0, 2, "Lines?", process)
add_button(0, 3, "Dummy", dummy_command, nothing)
add_button(0, 4, "Dummy", dummy_command, nothing)
add_button(0, 5, "Exit", done, nothing)
add_button(1, 0, "-Th", decrement_threshold, nothing)
add_button(1, 1, "+Th", increment_threshold, nothing)
add_button(1, 2, "-Len", decrement_min_line_length, nothing)
add_button(1, 3, "+Len", increment_min_line_length, nothing)
add_button(1, 4, "-Gap", decrement_max_line_gap, nothing)
add_button(1, 5, "+Gap", increment_max_line_gap, nothing)
message = add_message(1, 0, 6)
camera = cv2.VideoCapture(0)
width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = camera.get(cv2.CAP_PROP_FPS)
camera.release()
start_video(width, height, 2, 6)
