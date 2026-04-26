import cv2 as cv
import numpy as np

# ============================================================
# Shape detection using contours + approxPolyDP
# ============================================================

def show(name, img, w=800, h=600):
    """Show image in a resizable OpenCV window."""
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, w, h)
    cv.imshow(name, img)

# ------------------------------------------------------------
# 1) Load image
# ------------------------------------------------------------
img = cv.imread("images/shapes.png")
assert img is not None, "Error: shapes.png not found!"
img_draw = img.copy()

# ------------------------------------------------------------
# 2) Convert to grayscale + threshold (binary image)
#    White objects on black background works best.
# ------------------------------------------------------------
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(gray, 250, 255, cv.THRESH_BINARY)

show("1 - Threshold", thresh)

# ------------------------------------------------------------
# 3) Find contours
#    RETR_TREE: full hierarchy
#    CHAIN_APPROX_SIMPLE: compresses straight segments
# ------------------------------------------------------------
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# ------------------------------------------------------------
# 4) Filter contours: remove tiny noise and skip the largest one
#    Often the biggest contour is the image border/background.
# ------------------------------------------------------------
areas = [cv.contourArea(c) for c in contours]
if len(areas) == 0:
    raise RuntimeError("No contours found!")

background_id = int(np.argmax(areas))

# ------------------------------------------------------------
# 5) Loop through contours, approximate polygons, classify shape
# ------------------------------------------------------------
for idx, contour in enumerate(contours):

    # Skip background/largest contour
    if idx == background_id:
        continue

    area = cv.contourArea(contour)
    if area < 200:  # ignore very small contours (noise)
        continue

    # Polygon approximation (Douglas–Peucker)
    epsilon = 0.01 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)

    # Draw contour in red
    cv.drawContours(img_draw, [contour], -1, (0, 255, 255), 3)

    # Compute centroid using moments (for label placement)
    M = cv.moments(contour)
    if M["m00"] == 0:
        continue

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Classify by number of vertices
    v = len(approx)
    if v == 3:
        label = "Triangle"
    elif v == 4:
        label = "Quadrilateral"
    elif v == 5:
        label = "Pentagon"
    elif v == 6:
        label = "Hexagon"
    else:
        label = "Circle"

    # Put label 
    cv.putText(img_draw, label, (cx - 150, cy),
               cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 10, cv.LINE_AA)

# ------------------------------------------------------------
# 6) Show final result
# ------------------------------------------------------------
show("2 - Detected Shapes", img_draw)
cv.waitKey(0)
cv.destroyAllWindows()