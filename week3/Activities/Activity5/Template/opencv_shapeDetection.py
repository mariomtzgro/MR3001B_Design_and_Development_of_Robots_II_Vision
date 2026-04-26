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
# 6) Show final result
# ------------------------------------------------------------
show("2 - Detected Shapes", img_draw)
cv.waitKey(0)
cv.destroyAllWindows()