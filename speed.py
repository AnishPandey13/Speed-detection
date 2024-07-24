import cv2
import numpy as np
import time

# Load the Haar cascade classifier
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

# Start video capture
cap = cv2.VideoCapture('koteshwor video.mp4')  # Replace with your video file path

# Parameters for speed calculation
previous_frame_detections = []
distance_between_cars = 50  # Distance between cars in meters (adjust as needed)
frame_rate = cap.get(cv2.CAP_PROP_FPS)

def non_max_suppression(boxes, overlap_thresh):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    return boxes[pick].astype("int")

def calculate_speed(distance, time_diff):
    speed = distance / time_diff  # Speed in m/s
    speed_kmh = speed * 3.6  # Convert to km/h
    return speed_kmh

while True:
    # Read the frame
    ret, frame = cap.read()
    
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    current_frame_detections = []

    for (x, y, w, h) in cars:
        current_frame_detections.append((x, y, x+w, y+h))

    boxes = non_max_suppression(current_frame_detections, 0.3)

    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Calculate the position of the car
        current_position = (x1 + x2) / 2

        # Calculate time difference
        current_time = time.time()

        if previous_frame_detections:
            # Find the closest match in the previous frame
            closest_match = min(previous_frame_detections, key=lambda p: abs(p[0] - current_position))
            previous_position = closest_match[0]
            time_diff = current_time - closest_match[1]

            if time_diff > 0:
                # Calculate speed
                distance_covered = abs(current_position - previous_position)
                speed = calculate_speed(distance_covered, time_diff)
                cv2.putText(frame, f"Speed: {speed:.2f} km/h", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        previous_frame_detections = [(current_position, current_time)]

    # Display the frame
    cv2.imshow('Vehicle Speed Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
