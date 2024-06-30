import cv2
import time

# --- Configuration ---
stream_url = "rtsp://admin:panamet0@192.168.0.84:554/Streaming/Channels/102"  #channel 1 , substream 02
window_name = "Camera Stream"
buffer_size = 4 #increased buffer size 

# --- Connect to the Camera ---
cap = cv2.VideoCapture(stream_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
time.sleep(2) #increase sleep time to load 

# --- Check if Connection is Successful ---
if not cap.isOpened():
    print("Error: Could not open camera stream.")
    exit()

# --- Main Display Loop ---
while True:
    # Read a frame from the stream
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from stream. Reconnecting...")
        cap.release()
        cap = cv2.VideoCapture(stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        time.sleep(2) # give the reconnection some time
        continue

    # Display the frame in the window
    cv2.imshow(window_name, frame)
    
    # Check for key press to exit
    key = cv2.waitKey(1)
    if key == ord('q'):  # Press 'q' to quit
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
