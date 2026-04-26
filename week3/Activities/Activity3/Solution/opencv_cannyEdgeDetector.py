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
# 2) Convert to Grayscale
#    Canny operates on single-channel intensity images
# ------------------------------------------------------------
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
show("2 - Grayscale Image", gray)

# ------------------------------------------------------------
# 3) Gaussian Blur (Noise Reduction)
#    This step is critical for Canny:
#    - Reduces noise
#    - Prevents false edges
# ------------------------------------------------------------
blurred = cv.GaussianBlur(gray, (5, 5), 1.4)
show("3 - Gaussian Blurred", blurred)

# ------------------------------------------------------------
# 4) Canny Edge Detection
#    threshold1 = lower hysteresis threshold
#    threshold2 = upper hysteresis threshold
# ------------------------------------------------------------
edges = cv.Canny(
    blurred,
    threshold1=100,
    threshold2=200
)
show("4 - Canny Edges", edges)

# ------------------------------------------------------------
# 5) Overlay edges on the original image (optional, educational)
# ------------------------------------------------------------
overlay = img.copy()
overlay[edges > 0] = (0, 0, 255)  # draw edges in red
show("5 - Edges Overlay", overlay)

# ------------------------------------------------------------
# 6) Wait and close
# ------------------------------------------------------------
cv.waitKey(0)
cv.destroyAllWindows()