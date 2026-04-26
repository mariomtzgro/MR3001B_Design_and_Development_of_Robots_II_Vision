import cv2 as cv
import numpy as np

# ============================================================
# Hough Line Detection using Canny + HoughLinesP (OpenCV only)
# ============================================================

def show(win_name, image, w=800, h=600):
    """Display an image in a resizable OpenCV window."""
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(win_name, w, h)
    cv.imshow(win_name, image)

# ------------------------------------------------------------
# 1) Load image
# ------------------------------------------------------------
img = cv.imread('images/Puzzlebot_logo.png')
assert img is not None, "Image not found!"

show("1 - Original Image", img)

# ------------------------------------------------------------
# 2) Convert to grayscale
# ------------------------------------------------------------
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
show("2 - Grayscale", gray)

# ------------------------------------------------------------
# 3) Gaussian blur (noise reduction)
# ------------------------------------------------------------
blurred = cv.GaussianBlur(gray, (5, 5), 0)
show("3 - Gaussian Blurred", blurred)

# ------------------------------------------------------------
# 4) Canny edge detection
# ------------------------------------------------------------
edges = cv.Canny(blurred, threshold1=50, threshold2=150)
show("4 - Canny Edges", edges)

# ------------------------------------------------------------
# 5) Hough Line Detection (Probabilistic)
# ------------------------------------------------------------
lines = cv.HoughLinesP(
    edges,
    rho=1,                     # distance resolution in pixels
    theta=np.pi / 180,         # angle resolution in radians
    threshold=50,              # minimum votes
    minLineLength=10,          # minimum line length
    maxLineGap=10              # maximum gap between line segments
)

# ------------------------------------------------------------
# 6) Draw detected lines
# ------------------------------------------------------------
output = img.copy()

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(output, (x1, y1), (x2, y2), (0, 255, 0), 3)

show("5 - Hough Line Detection", output)

# ------------------------------------------------------------
# 7) Wait and cleanup
# ------------------------------------------------------------
cv.waitKey(0)
cv.destroyAllWindows()