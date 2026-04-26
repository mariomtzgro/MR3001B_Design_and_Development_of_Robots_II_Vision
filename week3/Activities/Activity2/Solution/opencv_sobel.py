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
# 2) Compute Sobel derivatives (floating-point)
#    gx = dI/dx  -> detects vertical edges (changes left-right)
#    gy = dI/dy  -> detects horizontal edges (changes up-down)
# ------------------------------------------------------------
gx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
gy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

# ------------------------------------------------------------
# 3) Prepare derivative images for DISPLAY
#    - convertScaleAbs() takes absolute value + converts to uint8
#    - Good for visualization, not for precise measurements
# ------------------------------------------------------------
gx_8u = cv.convertScaleAbs(gx)
gy_8u = cv.convertScaleAbs(gy)

show("3 - Sobel |Gx| (convertScaleAbs)", gx_8u)
show("4 - Sobel |Gy| (convertScaleAbs)", gy_8u)

# ------------------------------------------------------------
# 4) Combine gradients into a single edge-strength image
#    Gradient magnitude:
#      |grad| = sqrt(gx^2 + gy^2)
#    This produces a float image (CV_64F).
# ------------------------------------------------------------
mag = cv.magnitude(gx, gy)

# ------------------------------------------------------------
# 5) Normalize magnitude to 0..255 (uint8) for easy visualization
#    Without normalization, values may look almost black.
# ------------------------------------------------------------
mag_8u = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
show("5 - Gradient Magnitude (normalized)", mag_8u)

# ------------------------------------------------------------
# 6) Threshold the normalized magnitude to get a binary edge map
#    - Pixels above TH become 255 (edge)
#    - Pixels below TH become 0   (non-edge)
# ------------------------------------------------------------
TH = 50  # try 30..120 depending on image contrast
_, edges = cv.threshold(mag_8u, TH, 255, cv.THRESH_BINARY)
show(f"6 - Binary Edges (TH={TH})", edges)

# ------------------------------------------------------------
# 7) (Optional) Overlay edges on original image (for teaching)
# ------------------------------------------------------------
overlay = img.copy()
overlay[edges > 0] = (0, 0, 255)  # mark edges in red
show("7 - Edges Overlay (red)", overlay)

# ------------------------------------------------------------
# 8) Wait for keypress and close windows
# ------------------------------------------------------------
cv.waitKey(0)
cv.destroyAllWindows()