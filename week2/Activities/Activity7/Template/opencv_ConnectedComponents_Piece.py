import cv2 as cv
import numpy as np

# ============================================================
# Helper: display images nicely (resizable windows)
# ============================================================
def show(win_name, image, w=640, h=480):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(win_name, w, h)
    cv.imshow(win_name, image)




# ============================================================
# 1) Load image
# ============================================================
img = cv.imread("images/motor_holder/motor_holder_8.jpg")
assert img is not None, "Error: image not found!"

img_draw = img.copy()  # we will draw results on this image
show("1 - Original Image", img)


# ============================================================
# 11) Final visualization
# ============================================================
show("11 - Final (centers + distances)", img_draw)
cv.waitKey(0)
cv.destroyAllWindows()
