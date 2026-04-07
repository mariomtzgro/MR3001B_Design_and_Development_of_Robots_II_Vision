import cv2 as cv
import numpy as np

#HSV Values for Red Colour 


#HSV Values for Blue Colour 


if __name__=="__main__":
    #cap = cv.VideoCapture(2)
    video_path = 'videos/sample_video.mp4'
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print("Cannot open camera")

    #HSV Values for Green Colour
    while True:  # Reading frame from video or webcam
        
        ret, image = cap.read()  # if frame is read correctly ret is True
        
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break   
        
        k = cv.waitKey(25) & 0xFF
        if k == 27:
            break
        


        #Full red mask applied to te image
        cv.imshow('image', image)


    cap.release()
    cv.destroyAllWindows()
