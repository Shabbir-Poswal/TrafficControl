import cv2

def resize_video(video_path, output_path):
    # Open the input video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Unable to open the video file.")
        return

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the output frame size (640x640)
    output_width, output_height = 640, 640

    # Define the codec for the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Create a VideoWriter object to write the resized frames to a new video file
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

    # Loop through each frame of the video
    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if ret:
            # Resize the frame to the desired size (640x640)
            resized_frame = cv2.resize(frame, (output_width, output_height))

            # Write the resized frame to the output video file
            out.write(resized_frame)
        else:
            break

    # Release the video capture and video writer objects
    cap.release()
    out.release()

    print("Video resized successfully.")

# Specify the path to the input video file
input_video_path = 'sampleData/sample_video.mp4'

# Specify the path to the output video file
output_video_path = 'resized_video.mp4'

# Call the function to resize the video
resize_video(input_video_path, output_video_path)
