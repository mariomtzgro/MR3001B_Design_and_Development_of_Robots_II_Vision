import cv2 as cv
import numpy as np

# ============================================================
# Sobel Edge Detection + Thresholding (OpenCV only)
# - No matplotlib
# - Shows each step in resizable windows
# ============================================================

def show(win_name, image, w=640, h=480):
    """Helper to show images in a resizable window."""
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(win_name, w, h)
    cv.imshow(win_name, image)

# ------------------------------------------------------------
# 1) Load image and convert to grayscale
# ------------------------------------------------------------
img = cv.imread("images/MCR2_Logo_Black.png")
assert img is not None, "Error: image not found!"

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

show("1 - Original (BGR)", img)
show("2 - Grayscale", gray)



# ------------------------------------------------------------
# 8) Wait for keypress and close windows
# ------------------------------------------------------------
cv.waitKey(0)
cv.destroyAllWindows()