import cv2 as cv
import numpy as np

# Load image (OpenCV loads in BGR)
img = cv.imread("images/MCR2_Logo_Black.png")

if img is None:
    print("Failed to load image")
    exit()



cv.waitKey(0)
cv.destroyAllWindows()