import cv2 as cv
import numpy as np

# ============================================================
# Hough Circle Detection using OpenCV
# ============================================================

def show(win_name, image, w=640, h=640):
    """Display an image in a resizable OpenCV window."""
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(win_name, w, h)
    cv.imshow(win_name, image)

# ------------------------------------------------------------
# 1) Load image
# ------------------------------------------------------------
img = cv.imread('images/Puzzlebot_logo.png')
assert img is not None, "Error: image not found!"

show("1 - Original Image", img)

# ------------------------------------------------------------
# 2) Convert to Grayscale
#    HoughCircles works on single-channel images
# ------------------------------------------------------------
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
show("2 - Grayscale", gray)

# ------------------------------------------------------------
# 3) Gaussian Blur (Noise Reduction)
#    Smooths edges and improves circle detection stability
# ------------------------------------------------------------
blurred = cv.GaussianBlur(gray, (9, 9), 2)
show("3 - Gaussian Blurred", blurred)

# ------------------------------------------------------------
# 4) Hough Circle Detection
# ------------------------------------------------------------
circles = cv.HoughCircles(
    blurred,
    cv.HOUGH_GRADIENT,
    dp=1.9,           # Inverse resolution of accumulator
    minDist=60,       # Minimum distance between detected centers
    param1=200,       # Upper threshold for internal Canny
    param2=63,        # Accumulator threshold (higher = fewer circles)
    minRadius=0,      # Minimum circle radius
    maxRadius=150     # Maximum circle radius
)

# ------------------------------------------------------------
# 5) Draw detected circles
# ------------------------------------------------------------
output = img.copy()

if circles is not None:
    circles = np.uint16(np.around(circles))

    for (x, y, r) in circles[0]:
        # Draw outer circle (green)
        cv.circle(output, (x, y), r, (0, 255, 0), 4)

        # Draw center point (red)
        cv.circle(output, (x, y), 3, (0, 0, 255), -1)

show("4 - Detected Circles", output)

# ------------------------------------------------------------
# 6) Wait and cleanup
# ------------------------------------------------------------
cv.waitKey(0)
cv.destroyAllWindows()