# Import required libraries
import numpy as np  # (Not used in this code, but useful for image processing)
import cv2  # OpenCV library for computer vision tasks

# Initialize the camera (webcam)
# Argument '2' specifies the camera index (can be 0, 1, 2... depending on available cameras)
cam = cv2.VideoCapture(2)

# Counter for saved images
img_counter = 0

# Start an infinite loop to capture frames from the camera
while True:
    # Read a frame from the camera
    ret, frame = cam.read()

    # If the frame was not successfully captured, exit the loop
    if not ret:
        print("failed to grab frame")
        break

    # Display the current frame in a window titled 'capture'
    cv2.imshow("capture", frame)

    # Wait for a key press for 1 millisecond
    k = cv2.waitKey(1)

    # If the ESC key is pressed (ASCII code 27)
    if k % 256 == 27:
        print("Escape hit, closing...")  # Notify user
        break  # Exit the loop

    # If the SPACE key is pressed (ASCII code 32)
    elif k % 256 == 32:
        # Define the filename with the current counter value
        img_name = "./images/image_{}.png".format(img_counter)

        # Save the current frame to the specified file
        cv2.imwrite(img_name, frame)

        # Notify the user that the image was written
        print("{} written!".format(img_name))

        # Increment the counter for the next image
        img_counter += 1

# Release the camera resource after the loop ends
cam.release()

# Close all OpenCV windows
cv2.destroyAllWindows()