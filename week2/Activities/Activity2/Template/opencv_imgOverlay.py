import numpy as np
import cv2 as cv

# ---------------------------------------------------------------
# Load Images
# ---------------------------------------------------------------

# Base image: Puzzlebot robot render
img1 = cv.imread('images/MCR2__Puzzlebot_Jetson_Ed.png')

# Logo image to overlay on img1
img2 = cv.imread('images/Puzzlebot_logo.png')

# Resize img2 (logo) to desired dimensions (width=600px, height=300px)
img2 = cv.resize(img2, (600, 300))

# ---------------------------------------------------------------
# Validate if images are loaded correctly
# ---------------------------------------------------------------

assert img1 is not None, "Error: Base image (img1) could not be read."
assert img2 is not None, "Error: Logo image (img2) could not be read."

# ---------------------------------------------------------------
# Show Original Images
# ---------------------------------------------------------------

cv.namedWindow('Puzzlebot Robot', cv.WINDOW_NORMAL)
cv.imshow('Puzzlebot Robot', img1)

cv.namedWindow('Puzzlebot Logo', cv.WINDOW_NORMAL)
cv.imshow('Puzzlebot Logo', img2)


# ---------------------------------------------------------------
# Write Your Code Here
# ---------------------------------------------------------------


cv.waitKey(0)
cv.destroyAllWindows()