import numpy as np
import cv2 as cv

#Change parameter 0, ... to access other device
cap = cv.VideoCapture(2)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv.imshow('frame', gray)

    
# When everything done, release the captureq
cap.release()
cv.destroyAllWindows()