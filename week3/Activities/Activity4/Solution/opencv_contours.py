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
# 2) Convert to Grayscale
# ------------------------------------------------------------
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
show("2 - Grayscale", gray)

# ------------------------------------------------------------
# 3) Noise reduction (Gaussian Blur)
# ------------------------------------------------------------
blurred = cv.GaussianBlur(gray, (5, 5), 1.4)
show("3 - Gaussian Blurred", blurred)

# ------------------------------------------------------------
# 4) Edge Detection (Canny)
# ------------------------------------------------------------
edges = cv.Canny(blurred, threshold1=100, threshold2=200)
show("4 - Canny Edges", edges)

# ------------------------------------------------------------
# 5) Find Contours
#    - RETR_TREE: retrieves full hierarchy
#    - CHAIN_APPROX_SIMPLE: compresses straight segments
# ------------------------------------------------------------
contours, hierarchy = cv.findContours(
    edges,
    cv.RETR_TREE,
    cv.CHAIN_APPROX_SIMPLE
)

# ------------------------------------------------------------
# 6) Draw Contours
# ------------------------------------------------------------
output = img.copy()
cv.drawContours(output, contours, -1, (0, 255, 0), 2)
show("5 - Contours Detected", output)

# ------------------------------------------------------------
# 7) Wait and clean up
# ------------------------------------------------------------
cv.waitKey(0)
cv.destroyAllWindows()