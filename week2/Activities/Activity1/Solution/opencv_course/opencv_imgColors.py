import cv2 as cv
import numpy as np

# Load image (OpenCV loads in BGR)
img = cv.imread("images/MCR2_Logo_Black.png")

if img is None:
    print("Failed to load image")
    exit()

# Convert to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Convert to Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Convert BGR to LAB
lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)

# Display each version in its own window
cv.imshow("Original (BGR)", img)
cv.imshow("HSV", hsv)
cv.imshow("Grayscale", gray)
cv.imshow("LAB", lab)

cv.waitKey(0)
cv.destroyAllWindows()