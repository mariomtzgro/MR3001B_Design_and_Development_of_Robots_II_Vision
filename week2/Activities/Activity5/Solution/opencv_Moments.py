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
# 2. Convert to grayscale
# ---------------------------------------------------------------
# Moments are usually computed on binary or grayscale images
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------
# 3. Binary thresholding
# ---------------------------------------------------------------
# Convert image to binary:
# - Objects become white (255)
# - Background becomes black (0)
# THRESH_BINARY_INV is used to detect dark shapes on light background
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)


# ---------------------------------------------------------------
# 4. Compute raw spatial moments
# ---------------------------------------------------------------
# Moments describe the spatial distribution of white pixels
moments = cv2.moments(thresh)


# ---------------------------------------------------------------
# 5. Compute centroid from moments
# ---------------------------------------------------------------
# Centroid formulas:
# cx = m10 / m00
# cy = m01 / m00
# m00 corresponds to the object area
if moments["m00"] != 0:
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
else:
    cx, cy = 0, 0  # Safety check (no object detected)


# ---------------------------------------------------------------
# 6. Draw centroid and reference axes
# ---------------------------------------------------------------
# Copy original image for visualization
img_with_centroid = img.copy()

# Draw centroid (red dot)
cv2.circle(img_with_centroid, (cx, cy), 5, (0, 0, 255), -1)

# Draw reference lines to visualize centroid position
# Vertical line (Y-axis reference)
cv2.line(img_with_centroid, (cx, 0), (cx, cy), (255, 0, 0), 1)

# Horizontal line (X-axis reference)
cv2.line(img_with_centroid, (0, cy), (cx, cy), (255, 0, 0), 1)

# Label centroid coordinates
cv2.putText(
    img_with_centroid,
    f"Centroid: ({cx}, {cy})",
    (cx + 10, cy - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (255, 0, 0),
    1
)


# ---------------------------------------------------------------
# 7. Display moment values on the image
# ---------------------------------------------------------------
# Show key raw moments:
# - m00: area
# - m10, m01: used to compute centroid
info_x = 350
info_y = 430
line_spacing = 20

moment_texts = [
    f"m00 (Area): {moments['m00']:.2e}",
    f"m10: {moments['m10']:.2e}",
    f"m01: {moments['m01']:.2e}"
]

for i, text in enumerate(moment_texts):
    cv2.putText(
        img_with_centroid,
        text,
        (info_x, info_y + i * line_spacing),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1
    )


# ---------------------------------------------------------------
# 8. Display results using OpenCV windows
# ---------------------------------------------------------------
cv2.imshow("Original Image + Centroid + Moments", img_with_centroid)
cv2.imshow("Thresholded Image", thresh)


# ---------------------------------------------------------------
# 9. Print moments to terminal (for inspection)
# ---------------------------------------------------------------
print("\n--- Raw Spatial Moments ---")
print(f"m00 (Area): {moments['m00']}")
print(f"m10: {moments['m10']}")
print(f"m01: {moments['m01']}")

print("\n--- Central Moments (Shape-related) ---")
print(f"mu20: {moments['mu20']}")
print(f"mu02: {moments['mu02']}")
print(f"mu11: {moments['mu11']}")


# ---------------------------------------------------------------
# 10. Wait and clean up
# ---------------------------------------------------------------
cv2.waitKey(0)
cv2.destroyAllWindows()