import cv2
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict, deque
import time
import os
from datetime import datetime

print("OpenCV version:", cv2.__version__)


class CentroidTracker:
    # Este valor es el número de frames que un objeto puede desaparecer antes de ser eliminado
    def __init__(self, maxDisappeared=1):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.objectDirections = {}
        self.previousCentroids = {}
        self.directionDisplayCounter = {}
        self.printedDirections = set()

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.objectDirections[self.nextObjectID] = None
        self.previousCentroids[self.nextObjectID] = centroid
        self.directionDisplayCounter[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.objectDirections[objectID]
        del self.previousCentroids[objectID]
        del self.directionDisplayCounter[objectID]
        if objectID in self.printedDirections:
            self.printedDirections.remove(objectID)

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects, self.objectDirections, self.directionDisplayCounter

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                previousCentroid = self.previousCentroids[objectID]
                currentCentroid = inputCentroids[col]
                self.previousCentroids[objectID] = currentCentroid

                if previousCentroid[1] >= exit_line and currentCentroid[1] < exit_line:
                    self.objectDirections[objectID] = "Sale"
                    # Show direction for 30 frames
                    self.directionDisplayCounter[objectID] = 30
                    if objectID not in self.printedDirections:
                        print(f"ID {objectID} Sale")
                        self.printedDirections.add(objectID)
                    # Deregister immediately after exiting
                    self.deregister(objectID)
                elif previousCentroid[1] <= entry_line and currentCentroid[1] > entry_line:
                    self.objectDirections[objectID] = "Entra"
                    # Show direction for 30 frames
                    self.directionDisplayCounter[objectID] = 30
                    if objectID not in self.printedDirections:
                        print(f"ID {objectID} Entra")
                        self.printedDirections.add(objectID)

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects, self.objectDirections, self.directionDisplayCounter


# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Start video capture
cap = cv2.VideoCapture(
    "rtsp://admin:panamet0@192.168.0.84:554/Streaming/Channels/102q")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Increase buffer size

# Print frame shape and frame rate
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f"Frame shape: {frame.shape}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Frame rate: {fps}")

        # Save the first frame as a PNG file
        print("Saving first frame as first_frame.png")
        cv2.imwrite("first_frame.png", frame)

# Initialize centroid tracker
ct = CentroidTracker(maxDisappeared=40)
(H, W) = (None, None)

# Define entry and exit zones
entry_line = int(60 / 4)  # Blue line
exit_line = int(70 / 4)  # Red line

# Define the no-zone mask as a polygon
no_zone = np.array([[int(0 / 4), int(174 / 4)], [int(203 / 4), int(275 / 4)], [int(398 / 4), int(37 / 4)],
                   [int(365 / 4), int(22 / 4)], [int(239 / 4), int(14 / 4)], [int(0 / 4), int(72 / 4)], [int(0 / 4), int(174 / 4)]])

# Initialize video recording parameters
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 format
recording_duration = 3  # seconds before and after the event
# Buffer for 3 seconds before and 3 seconds after
buffer_size = int(fps * recording_duration * 2)
frame_buffer = deque(maxlen=buffer_size)

# Create the events directory if it doesn't exist
os.makedirs("./events", exist_ok=True)

# Function to save video event


def save_event(event_type, frame_buffer, timestamp):
    filename = f'./events/{event_type}_{timestamp}.mp4'
    out = cv2.VideoWriter(filename, fourcc, fps, (W, H))
    for frame in frame_buffer:
        out.write(frame)
    out.release()

# Function to continue recording for a specified duration


def continue_recording(duration):
    end_time = time.time() + duration
    while time.time() < end_time:
        ret, frame = cap.read()
        if not ret:
            break
        frame_buffer.append(frame.copy())
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Flag to avoid saving multiple events simultaneously
is_saving_event = False
current_event = None
save_event = False

# Start processing the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame, retrying...")
        cap.release()
        cap = cv2.VideoCapture(
            "rtsp://admin:panamet0@192.168.0.84:554/Streaming/Channels/102q")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Increase buffer size
        continue

    # Resize frame to ¼ of the original resolution
    frame = cv2.resize(frame, (160, 90))

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # Add the current frame to the buffer
    frame_buffer.append(frame.copy())

    # Object detection
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # Car detection
            if confidence > 0.5 and class_id == classes.index("car"):
                center_x = int(detection[0] * W)
                center_y = int(detection[1] * H)
                w = int(detection[2] * W)
                h = int(detection[3] * H)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Check if the center is within the no-zone mask
                if cv2.pointPolygonTest(no_zone, (center_x, center_y), False) < 0:
                    continue  # Ignore this detection if it is outside the no-zone

                boxes.append((x, y, x + w, y + h))
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indexes) > 0:
        indexes = indexes.flatten()

    rects = []
    for i in indexes:
        rects.append(boxes[i])

    objects, directions, display_counters = ct.update(rects)

    for (objectID, centroid) in objects.items():
        text = f"ID {objectID}"
        cv2.putText(frame, text, (centroid[0] + 10, centroid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
        cv2.circle(frame, (centroid[0], centroid[1]), 1, (0, 255, 0), -1)

        # Display the direction and save video events
        direction = directions.get(objectID, None)
        if display_counters[objectID] > 0:
            cv2.putText(frame, f"ID {objectID} {direction}", (90, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255) if direction == "Sale" else (255, 0, 0), 1)
            display_counters[objectID] -= 1

    # Draw the lines on the frame
    cv2.line(frame, (0, exit_line), (W, exit_line),
             (0, 0, 255), 1)  # Red line for exit
    cv2.line(frame, (0, entry_line), (W, entry_line),
             (0, 255, 0), 1)  # Green line for entry

    # Create a mask for the no-zone
    mask = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.fillPoly(mask, [no_zone], (255, 0, 0))  # Blue color

    # Create an alpha mask for blending
    alpha_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(alpha_mask, [no_zone], 255)

    # Combine the frame and the mask using the alpha mask
    frame = cv2.addWeighted(frame, 1, mask, 0.8, 0, frame)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
