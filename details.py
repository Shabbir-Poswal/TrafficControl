import cv2

def get_video_details(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Unable to open the video file.")
        return

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get the width and height of the video frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get the codec name of the video file
    codec = cap.get(cv2.CAP_PROP_FOURCC)

    # Print the details of the video
    print("Video Details:")
    print("--------------")
    print(f"Total Frames: {total_frames}")
    print(f"Frame Rate: {fps} frames per second")
    print(f"Frame Size: {width}x{height}")
    print(f"Codec: {codec}")

    # Release the video capture object
    cap.release()

# Specify the path to the video file
video_path = '4K Video of Highway Traffic!.mp4'

# Call the function to get details about the video
get_video_details(video_path)
