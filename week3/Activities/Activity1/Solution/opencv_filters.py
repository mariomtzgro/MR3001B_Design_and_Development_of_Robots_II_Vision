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
# 3) APPLY FILTERS
# ============================================================
noisy_sp = add_salt_pepper_noise(img, prob=0.02)
show("Salt & Pepper Noise", noisy_sp)

noisy_gaussian = add_gaussian_noise(img)
show("Gaussian Noise", noisy_gaussian)

# ---------- Sharpening Filter ----------
sharpen_kernel = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
])

sharpened = cv.filter2D(img, -1, sharpen_kernel)
show("Sharpened Image", sharpened)


# ---------- Median Filter (for Salt & Pepper) ----------
median_denoised = cv.medianBlur(noisy_sp, 5)
show("Median Filter (Salt & Pepper Removed)", median_denoised)


# ---------- Gaussian Blur (for Gaussian Noise) ----------
gaussian_blur = cv.GaussianBlur(noisy_gaussian, (5, 5), 0)
show("Gaussian Blur (Gaussian Noise)", gaussian_blur)


# ---------- Averaging Filter (Mean Filter) ----------
average_blur = cv.blur(noisy_gaussian, (11, 11))
show("Average Blur (Gaussian Noise)", average_blur)

# ============================================================
# 4) Wait and clean up
# ============================================================
cv.waitKey(0)
cv.destroyAllWindows()