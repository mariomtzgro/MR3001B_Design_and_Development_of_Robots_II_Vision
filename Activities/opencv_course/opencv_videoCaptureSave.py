import numpy as np
import cv2 as cv

# Open the camera device (use index 2, can be 0 or 1 depending on your system)
cap = cv.VideoCapture(2)

# Define the codec and prepare to create VideoWriter object later (but don't start recording yet)
fourcc = cv.VideoWriter_fourcc(*'MJPG')
# Create a new VideoWriter object (you can add timestamp to avoid overwriting)
out = cv.VideoWriter('./videos/Test.avi', fourcc, 30.0, (640, 480))

# Boolean flag to track whether recording is active
recording = False

while cap.isOpened():
    # Read a frame from the camera
    ret, frame = cap.read()

    # If the frame wasn't read successfully, exit the loop
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # If recording is ON, write the current frame to the video file
    if recording:
        out.write(frame)

    # Display the current frame in a window named 'frame'
    cv.imshow('frame', frame)

    # Listen for keyboard events (check every 1 ms)
    k = cv.waitKey(1) & 0xFF
    # If the 'q' key is pressed, exit the loop
    if k == ord('q'):
        print("Quitting...")
        break

    # If the SPACEBAR is pressed, toggle recording ON/OFF
    elif k == ord(' '):  # SPACE key
        recording = not recording  # Toggle the recording state
        if recording:
            # Start recording: create the VideoWriter object
            print("Recording started...")

        else:
            # Stop recording: release the VideoWriter object
            print("Recording stopped.")
            break

# Release the camera resource when the loop ends
cap.release()
# If recording was still on when quitting, release the writer
out.release()
# Close all OpenCV windows
cv.destroyAllWindows()