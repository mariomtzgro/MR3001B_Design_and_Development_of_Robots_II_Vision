import cv2 as cv
import numpy as np

# ============================================================
# Canny Edge Detection – Step by Step (OpenCV only)
# ============================================================

def show(win_name, image, w=640, h=480):
    """Helper function to show resizable OpenCV windows."""
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(win_name, w, h)
    cv.imshow(win_name, image)

# ------------------------------------------------------------
# 1) Load image
# ------------------------------------------------------------
img = cv.imread('images/MCR2_Logo_Black.png')
assert img is not None, "Error: Image not found!"

show("1 - Original Image", img)



# ------------------------------------------------------------
# 6) Wait and close
# ------------------------------------------------------------
cv.waitKey(0)
cv.destroyAllWindows()