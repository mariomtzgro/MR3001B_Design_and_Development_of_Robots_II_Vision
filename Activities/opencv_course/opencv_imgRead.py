import cv2 as cv

#Read the image in the following path "images/MCR2_Logo_Black.png"
img = cv.imread('images/MCR2_Logo_Black.png')
#If no image is present, finish the program
if img is None:
    print("Failed to load image")
    exit()
#Show the image on a pop-up window
cv.imshow('Example Image', img)
#Wait for any key to be pressed
cv.waitKey(0)
#Destroy all windows
cv.destroyAllWindows()