import cv2
import os
from datetime import datetime

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Start video capture
cap = cv2.VideoCapture("rtsp://admin:panamet0@192.168.0.84:554/Streaming/Channels/102")

# Check if the video capture is opened
if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

# Get the width, height, and frames per second (fps) of the video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create the output directory if it doesn't exist
os.makedirs("./recordings", exist_ok=True)

# Initialize video recording parameters
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 format
segment_duration = 60 * 60  # 1 hour in seconds

# Function to start a new video writer
def start_new_video_writer():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'./recordings/recording_{timestamp}.mp4'
    return cv2.VideoWriter(filename, fourcc, fps, (width, height))

# Start the first video writer
out = start_new_video_writer()
start_time = datetime.now()

# Read the first frame to print its shape and save it as PNG
ret, frame = cap.read()
if ret:
    print("Shape of the first frame:", frame.shape)
    # Save the first frame as a PNG image
    cv2.imwrite('./recordings/first_frame.png', frame)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Write the frame to the current video file
    out.write(frame)

    # Check if one hour has passed to start a new video file
    elapsed_time = (datetime.now() - start_time).total_seconds()
    if elapsed_time >= segment_duration:
        out.release()
        out = start_new_video_writer()
        start_time = datetime.now()

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
