import numpy as np
import cv2 as cv

# ============================================================
# Helper: show images in resizable windows
# ============================================================
def show(name, img, w=640, h=480):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, w, h)
    cv.imshow(name, img)

# ============================================================
# 1) Load image
# ============================================================
img = cv.imread('images/Puzzlebot_hand.png')
assert img is not None, "Image not found!"

show("Original Image", img)

# ============================================================
# 2) CREATE NOISY IMAGES (inputs for filters)
# ============================================================
# ---------- Salt & Pepper Noise ----------
def add_salt_pepper_noise(image, prob=0.02):
    noisy = image.copy()
    h, w, c = image.shape
    num_pixels = int(prob * h * w)

    # Salt (white pixels)
    ys = np.random.randint(0, h, num_pixels // 2)
    xs = np.random.randint(0, w, num_pixels // 2)
    noisy[ys, xs] = [255, 255, 255]
    # Pepper (black pixels)
    ys = np.random.randint(0, h, num_pixels // 2)
    xs = np.random.randint(0, w, num_pixels // 2)
    noisy[ys, xs] = [0, 0, 0]

    return noisy

# ---------- Gaussian Noise ----------
def add_gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)



# ============================================================
# 4) Wait and clean up
# ============================================================
cv.waitKey(0)
cv.destroyAllWindows()