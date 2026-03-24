import cv2 as cv

# Read the video file (replace with your actual video file path)
video_path = 'videos/sample_video.mp4'
cap = cv.VideoCapture(video_path)

# Check if the video was successfully opened
if not cap.isOpened():
    print("Failed to load video")
    exit()

# Loop through each frame of the video
while True:
    # Read a frame
    ret, frame = cap.read()

    # If no frame is returned, the video has ended
    if not ret:
        print("End of video or failed to read frame")
        break

    # Display the frame in a window
    cv.imshow('Video Playback', frame)

    # Wait for 25ms and break if 'q' is pressed (adjust time to control playback speed)
    if cv.waitKey(25) & 0xFF == ord('q'):
        print("Playback interrupted by user")
        break

# Release the video capture object
cap.release()

# Destroy all OpenCV windows
cv.destroyAllWindows()