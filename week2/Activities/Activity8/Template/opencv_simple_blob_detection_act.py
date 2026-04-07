import cv2 as cv
import numpy as np


# Main Code Execution Starts Here
if __name__ == "__main__":

    # Start video capture
    # You can switch to a webcam by using cv.VideoCapture(0)
    video_path = 'videos/sample_video.mp4'
    cap = cv.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Cannot open camera or video file")


    # Step 11: Release resources and close windows when done
    cap.release()
    cv.destroyAllWindows()