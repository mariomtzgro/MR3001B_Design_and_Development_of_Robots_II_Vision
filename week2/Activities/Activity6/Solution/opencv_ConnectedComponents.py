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
# 2) Convert to grayscale
# ------------------------------------------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ------------------------------------------------------------
# 3) Binary thresholding
#    Invert so objects become white (foreground)
# ------------------------------------------------------------
_, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)

# ------------------------------------------------------------
# 4) Connected Component Analysis (4-connectivity)
# ------------------------------------------------------------
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    binary, connectivity=4
)

# ------------------------------------------------------------
# 5) Assign random colors to each component
# ------------------------------------------------------------
colored_components = np.zeros_like(img)

for label_id in range(1, num_labels):  # skip background (label 0)
    mask = (labels == label_id)
    color = np.random.randint(0, 255, size=3)
    colored_components[mask] = color

# ------------------------------------------------------------
# 6) Draw bounding boxes and centroids
# ------------------------------------------------------------
output = colored_components.copy()

for label_id in range(1, num_labels):

    x, y, w, h, area = stats[label_id]
    cx, cy = centroids[label_id]

    # Draw bounding box (green)
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw centroid (red)
    cv2.circle(output, (int(cx), int(cy)), 5, (0, 0, 255), -1)

    # Label centroid coordinates
    cv2.putText(
        output,
        f"({int(cx)}, {int(cy)})",
        (int(cx) + 10, int(cy)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

# ------------------------------------------------------------
# 7) Display all processing stages (OpenCV windows)
# ------------------------------------------------------------
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("Binary Threshold", cv2.WINDOW_NORMAL)
cv2.namedWindow("Connected Components (Colored)", cv2.WINDOW_NORMAL)
cv2.namedWindow("Bounding Boxes + Centroids", cv2.WINDOW_NORMAL)

cv2.resizeWindow("Original Image", 640, 480)
cv2.resizeWindow("Binary Threshold", 640, 480)
cv2.resizeWindow("Connected Components (Colored)", 640, 480)
cv2.resizeWindow("Bounding Boxes + Centroids", 640, 480)

cv2.imshow("Original Image", img)
cv2.imshow("Binary Threshold", binary)
cv2.imshow("Connected Components (Colored)", colored_components)
cv2.imshow("Bounding Boxes + Centroids", output)

cv2.waitKey(0)
cv2.destroyAllWindows()