import cv2
import numpy as np

# ---------------------------------------------------------------
# 1. Load image
# ---------------------------------------------------------------
# Read the image from disk (BGR format)
img = cv2.imread('images/figure1.png')
assert img is not None, "Error: Image not found!"

# Resize image to a fixed size for consistency
img = cv2.resize(img, (640, 480))





# ---------------------------------------------------------------
# 10. Wait and clean up
# ---------------------------------------------------------------
cv2.waitKey(0)
cv2.destroyAllWindows()