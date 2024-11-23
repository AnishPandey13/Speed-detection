import cv2
import numpy as np
import pandas as pd
import os
from tracker import *

# Load YOLOv3 model
model_config = "yolov3.cfg"
model_weights = "yolov3.weights"
net = cv2.dnn.readNet(model_weights, model_config)

# Load class labels
class_list = []
with open("coco.names", "r") as f:
    class_list = f.read().strip().split("\n")

tracker = Tracker()
count = 0
cap = cv2.VideoCapture("v.mp4")

# Get the video frame rate for accurate timing
fps = cap.get(cv2.CAP_PROP_FPS)

down = {}
up = {}
counter_down = []
counter_up = []

red_line_y = 198
blue_line_y = 268
offset = 6

# Create a folder to save frames
if not os.path.exists("detected_frames"):
    os.makedirs("detected_frames")

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.avi", fourcc, 20.0, (1020, 500))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    frame = cv2.resize(frame, (1020, 500))

    # Prepare input for YOLOv3
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(layer_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter by confidence and class
            if confidence > 0.5 and class_list[class_id] == "car":  # Adjusted threshold
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            detections.append([x, y, x + w, y + h])

    # Update tracker
    bbox_id = tracker.update(detections)

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int((x3 + x4) / 2)
        cy = int((y3 + y4) / 2)

        # Calculate speed when moving down
        if red_line_y < (cy + offset) and red_line_y > (cy - offset):
            down[id] = count / fps  # Time in seconds based on frame count
        if id in down:
            if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
                elapsed_time = (count / fps) - down[id]
                if counter_down.count(id) == 0:
                    counter_down.append(id)
                    distance = 10  # meters
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(a_speed_kh)) + " Km/h", (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        # Calculate speed when moving up
        if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
            up[id] = count / fps  # Time in seconds based on frame count
        if id in up:
            if red_line_y < (cy + offset) and red_line_y > (cy - offset):
                elapsed1_time = (count / fps) - up[id]
                if counter_up.count(id) == 0:
                    counter_up.append(id)
                    distance1 = 10  # meters
                    a_speed_ms1 = distance1 / elapsed1_time
                    a_speed_kh1 = a_speed_ms1 * 3.6
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(a_speed_kh1)) + " Km/h", (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    text_color = (0, 0, 0)
    yellow_color = (0, 255, 255)
    red_color = (0, 0, 255)
    blue_color = (255, 0, 0)

    cv2.rectangle(frame, (0, 0), (250, 90), yellow_color, -1)
    cv2.line(frame, (172, red_line_y), (774, red_line_y), red_color, 2)
    cv2.putText(frame, "Red Line", (172, red_line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.line(frame, (8, blue_line_y), (927, blue_line_y), blue_color, 2)
    cv2.putText(frame, "Blue Line", (8, blue_line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, "Going Down - " + str(len(counter_down)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, "Going Up - " + str(len(counter_up)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    frame_filename = f"detected_frames/frame_{count}.jpg"
    cv2.imwrite(frame_filename, frame)

    out.write(frame)
    cv2.imshow("frames", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
