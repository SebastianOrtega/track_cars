import cv2
import time

# Function to resize video and provide feedback on remaining time


def resize_video(input_file, output_file, new_width, new_height, target_fps):
    cap = cv2.VideoCapture(input_file)

    # Get the original video's properties
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, target_fps,
                          (new_width, new_height))

    frame_count = 0
    start_time = time.time()

    # Calculate frame interval to match target FPS
    frame_interval = int(original_fps / target_fps)
    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process only every nth frame where n is the frame interval
        if current_frame % frame_interval == 0:
            # Resize the frame using OpenCV
            resized_frame = cv2.resize(frame, (new_width, new_height))

            # Write the resized frame to the new video
            out.write(resized_frame)

            frame_count += 1

            # Calculate and print the remaining time and frames per second
            elapsed_time = time.time() - start_time
            avg_time_per_frame = elapsed_time / frame_count
            remaining_frames = (total_frames - frame_count *
                                frame_interval) // frame_interval
            remaining_time = remaining_frames * avg_time_per_frame

            print(
                f"Processed {frame_count}/{total_frames} frames. Estimated remaining time: {remaining_time:.2f} seconds. FPS: {1/avg_time_per_frame:.2f}", end='\r')

        current_frame += 1

    # Release resources
    cap.release()
    out.release()


# Example usage
input_video = "prueba_2.mp4"
output_video = "resized_video.mp4"
resize_video(input_video, output_video, 704, 408, 10)
