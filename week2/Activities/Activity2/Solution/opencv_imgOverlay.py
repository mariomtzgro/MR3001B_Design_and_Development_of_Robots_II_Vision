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
# Create ROI (Region of Interest) in the top-right corner of img1
# ---------------------------------------------------------------

rows1, cols1, channels1 = img1.shape   # Base image dimensions
rows, cols, channels = img2.shape      # Logo image dimensions

# Select ROI area from img1 where the logo will be placed
roi = img1[0:rows, cols1 - cols:cols1]

# ---------------------------------------------------------------
# Create Mask and Inverse Mask from the Logo
# ---------------------------------------------------------------

# Convert logo image to grayscale (for thresholding)
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# Threshold to create the mask (white logo area)
ret, mask = cv.threshold(img2_gray, 190, 255, cv.THRESH_BINARY)

# Invert mask (black logo area)
mask_inv = cv.bitwise_not(mask)

# Show mask
cv.namedWindow('Mask', cv.WINDOW_NORMAL)
cv.imshow('Mask', mask)
cv.namedWindow('Inverted Mask', cv.WINDOW_NORMAL)
cv.imshow('Inverted Mask', mask_inv)

# ---------------------------------------------------------------
# Black-out logo area on ROI and Extract the Logo from img2
# ---------------------------------------------------------------

# Background of ROI with logo area blacked out (using mask)
img1_bg = cv.bitwise_and(roi, roi, mask=mask)

cv.namedWindow('ROI with Logo Area Cleared', cv.WINDOW_NORMAL)
cv.imshow('ROI with Logo Area Cleared', img1_bg)

# Extract the logo from img2 (using inverse mask)
img2_fg = cv.bitwise_and(img2, img2, mask=mask_inv)

cv.namedWindow('Extracted Logo', cv.WINDOW_NORMAL)
cv.imshow('Extracted Logo', img2_fg)

# ---------------------------------------------------------------
# Combine Cleared ROI and Extracted Logo, then Update Base Image
# ---------------------------------------------------------------

# Add the cleared ROI and logo together to form final overlay
dst = cv.add(img1_bg, img2_fg)

cv.namedWindow('Combined ROI and Logo', cv.WINDOW_NORMAL)
cv.imshow('Combined ROI and Logo', dst)

# Replace the ROI in img1 with the new combined image
img1[0:rows, cols1 - cols:cols1] = dst

# ---------------------------------------------------------------
# Show Final Result
# ---------------------------------------------------------------

cv.namedWindow('Final Image with Logo', cv.WINDOW_NORMAL)
cv.imshow('Final Image with Logo', img1)

# ---------------------------------------------------------------
# Wait for keypress and close all windows
# ---------------------------------------------------------------

cv.waitKey(0)
cv.destroyAllWindows()