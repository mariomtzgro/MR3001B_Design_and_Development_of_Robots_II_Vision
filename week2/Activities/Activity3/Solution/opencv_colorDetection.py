import cv2 as cv
import numpy as np

#HSV Values for Red Colour 
H_red_low_limit = 0
S_red_low_limit = 95
V_red_low_limit = 130

H_red_up_limit = 5
S_red_up_limit = 255
V_red_up_limit = 255

#HSV Values for Blue Colour 
H_blue_low_limit = 95
S_blue_low_limit = 90
V_blue_low_limit = 90

H_blue_up_limit = 110
S_blue_up_limit = 255
V_blue_up_limit = 255

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
        
        # Convert BGR to HSV
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)    
        
        #Ranges
        red_lower_limit = np.array([H_red_low_limit, S_red_low_limit, V_red_low_limit])
        red_upper_limit = np.array([H_red_up_limit, S_red_up_limit, V_red_up_limit])
        
        blue_lower_limit = np.array([H_blue_low_limit, S_blue_low_limit, V_blue_low_limit])
        blue_upper_limit = np.array([H_blue_up_limit, S_blue_up_limit, V_blue_up_limit])
        
        #Masks
        red_mask = cv.inRange(hsv_image, red_lower_limit, red_upper_limit)
        blue_mask = cv.inRange(hsv_image, blue_lower_limit, blue_upper_limit)
        
        #Partial masks applied to the image (AND Function with the image)
        red_img_result = cv.bitwise_and(image,image,mask=red_mask)
        blue_img_result = cv.bitwise_and(image,image,mask=blue_mask)

        #Full red mask applied to te image
        cv.imshow('image', image)
        cv.imshow('red mask', red_img_result)
        cv.imshow('blue mask', blue_img_result)

    cap.release()
    cv.destroyAllWindows()
