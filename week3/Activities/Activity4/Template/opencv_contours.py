import cv2 as cv
import numpy as np

# ============================================================
# Contour Detection using Canny + findContours (OpenCV only)
# ============================================================

def show(win_name, image, w=800, h=600):
    """Display an image in a resizable OpenCV window."""
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(win_name, w, h)
    cv.imshow(win_name, image)

# ------------------------------------------------------------
# 1) Load image
# ------------------------------------------------------------
img = cv.imread('images/MCR2_Logo_Black.png')
assert img is not None, "Image not found!"

show("1 - Original Image", img)



# ------------------------------------------------------------
# 7) Wait and clean up
# ------------------------------------------------------------
cv.waitKey(0)
cv.destroyAllWindows()