import cv2
import numpy as np

# ------------------------------------------------------------
# 1) Load image
# ------------------------------------------------------------
img = cv2.imread("images/figure2.png")
assert img is not None, "Image not found!"

# Resize for consistent display
img = cv2.resize(img, (640, 480))


# ------------------------------------------------------------
# Write Your Code Here
# ------------------------------------------------------------

cv2.waitKey(0)
cv2.destroyAllWindows()