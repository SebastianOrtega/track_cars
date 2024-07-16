import cv2
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict, deque
import time
import os
from datetime import datetime

print("OpenCV version:", cv2.__version__)


class CentroidTracker:
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
                    self.objectDirections[objectID] = "Exit"
                    if objectID not in self.printedDirections:
                        print(f"ID {objectID} Exit")
                        self.printedDirections.add(objectID)
                    self.deregister(objectID)
                elif previousCentroid[1] <= entry_line and currentCentroid[1] > entry_line:
                    self.objectDirections[objectID] = "Entry"
                    if objectID not in self.printedDirections:
                        print(f"ID {objectID} Entry")
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
# cap = cv2.VideoCapture("rtsp://admin:panamet0@192.168.0.84:554/Streaming/Channels/102q")
cap = cv2.VideoCapture("./prueba2.mp4")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Increase buffer size

# Print frame shape and frame rate
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f"Frame shape: {frame.shape}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Frame rate: {fps}")

# Initialize centroid tracker
ct = CentroidTracker(maxDisappeared=40)
(H, W) = (None, None)

# Define entry and exit zones
entry_line = int(60 / 4)  # Blue line
exit_line = int(70 / 4)  # Red line

# Define the no-zone mask as a polygon
no_zone = np.array([[int(0 / 4), int(174 / 4)], [int(203 / 4), int(275 / 4)], [int(398 / 4), int(37 / 4)],
                    [int(365 / 4), int(22 / 4)], [int(239 / 4), int(14 / 4)], [int(0 / 4), int(72 / 4)], [int(0 / 4), int(174 / 4)]])

# Start processing the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame, retrying...")
        cap.release()
        cap = cv2.VideoCapture(
            "rtsp://admin:panamet0@192.168.0.84:554/Streaming/Channels/102")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Increase buffer size
        print("Reconnected to the camera")
        continue

    # Resize frame to ¼ of the original resolution
    frame = cv2.resize(frame, (160, 90))

    if W is None or H is None:
        (H, W) = frame.shape[:2]

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
            if confidence > 0.5 and class_id == classes.index("car"):
                center_x = int(detection[0] * W)
                center_y = int(detection[1] * H)
                w = int(detection[2] * W)
                h = int(detection[3] * H)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                if cv2.pointPolygonTest(no_zone, (center_x, center_y), False) < 0:
                    continue

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
        direction = directions.get(objectID, None)
        if display_counters[objectID] > 0:
            display_counters[objectID] -= 1

cap.release()
