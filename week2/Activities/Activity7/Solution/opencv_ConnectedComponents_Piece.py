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
# 0) Parameters (HSV thresholds + morphology + ROI)
# ============================================================

# --- Purple threshold to find the part (coarse) ---
H1_LOW, S1_LOW, V1_LOW = 120, 15, 15
H1_UP,  S1_UP,  V1_UP  = 160, 255, 255

# --- Purple threshold to refine inside ROI (slightly stricter V) ---
H2_LOW, S2_LOW, V2_LOW = 120, 15, 35
H2_UP,  S2_UP,  V2_UP  = 160, 255, 255

# Morphology settings
KERNEL_SIZE = (5, 5)
DILATE_IT_1 = 10
ERODE_IT_1  = 5

KERNEL_SIZE_2 = (5, 5)
ERODE_IT_2  = 5
DILATE_IT_2 = 5

# Circular ROI radius around the part centroid (pixels)
ROI_RADIUS = 550

# Connected component area filtering (heuristics)
MIN_AREA = 1000
MAX_AREA = 50000


# ============================================================
# 1) Load image
# ============================================================
img = cv.imread("images/motor_holder/motor_holder_8.jpg")
assert img is not None, "Error: image not found!"

img_draw = img.copy()  # we will draw results on this image
show("1 - Original Image", img)

# ============================================================
# 2) Convert to HSV (better for color segmentation)
# ============================================================
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# ============================================================
# 3) Step A: Detect the purple part (coarse mask)
# ============================================================
lower1 = np.array([H1_LOW, S1_LOW, V1_LOW])
upper1 = np.array([H1_UP,  S1_UP,  V1_UP])

purple_mask = cv.inRange(hsv, lower1, upper1)
show("2 - Purple Mask (raw)", purple_mask)

# ============================================================
# 4) Clean the purple mask (morphology)
#    Dilation fills small gaps, erosion removes noise.
# ============================================================
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, KERNEL_SIZE)

purple_mask_clean = cv.dilate(purple_mask, kernel, iterations=DILATE_IT_1)
purple_mask_clean = cv.erode(purple_mask_clean, kernel, iterations=ERODE_IT_1)
show("3 - Purple Mask (clean)", purple_mask_clean)

# Visualize masked result (what pixels are selected)
purple_part = cv.bitwise_and(img, img, mask=purple_mask_clean)
show("4 - Purple Part Extracted", purple_part)

# ============================================================
# 5) Compute centroid of the purple region using image moments
#    centroid = (m10/m00, m01/m00)
# ============================================================
M = cv.moments(purple_mask_clean)

if M["m00"] != 0:
    part_cx = int(M["m10"] / M["m00"])
    part_cy = int(M["m01"] / M["m00"])
else:
    part_cx, part_cy = 0, 0

# Draw centroid on the extracted part for visualization
purple_part_centroid = purple_part.copy()
cv.circle(purple_part_centroid, (part_cx, part_cy), 20, (0, 0, 255), -1)
cv.putText(purple_part_centroid, "Part centroid", (part_cx + 10, part_cy - 10),
           cv.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5)
show("5 - Part centroid (on purple part)", purple_part_centroid)

# ============================================================
# 6) Create a circular ROI around the part centroid
#    This limits the search region to avoid detecting unrelated blobs.
# ============================================================
H, W = img.shape[:2]
roi_mask = np.zeros((H, W), dtype=np.uint8)
cv.circle(roi_mask, (part_cx, part_cy), ROI_RADIUS, 255, -1)
show("6 - Circular ROI mask", roi_mask)

# Apply ROI mask to HSV image (we only keep pixels inside the circle)
hsv_roi = cv.bitwise_and(hsv, hsv, mask=roi_mask)

# ============================================================
# 7) Step B: Refine purple segmentation inside ROI (stricter)
# ============================================================
lower2 = np.array([H2_LOW, S2_LOW, V2_LOW])
upper2 = np.array([H2_UP,  S2_UP,  V2_UP])

roi_purple_mask = cv.inRange(hsv_roi, lower2, upper2)
show("7 - Purple Mask inside ROI (raw)", roi_purple_mask)

# Clean ROI mask (opening-ish: erosion then dilation)
kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, KERNEL_SIZE_2)
roi_purple_mask_clean = cv.erode(roi_purple_mask, kernel2, iterations=ERODE_IT_2)
roi_purple_mask_clean = cv.dilate(roi_purple_mask_clean, kernel2, iterations=DILATE_IT_2)
show("8 - Purple Mask inside ROI (clean)", roi_purple_mask_clean)

# ============================================================
# 8) Connected Components on the INVERTED mask
#    Why invert?
#    - If purple area is white, holes are black.
#    - Inverting makes holes white → easier to label as components.
# ============================================================
inv = cv.bitwise_not(roi_purple_mask_clean)
show("9 - Inverted mask (holes become white)", inv)

num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(inv, connectivity=8)

# Identify background label by largest area (area column is index 4)
areas = stats[:, 4]
background_label = int(np.argmax(areas))

# ============================================================
# 9) Filter components by area and collect centroids
# ============================================================
kept_centers = []

# Visual: color labels (optional, helps students)
labels_vis = cv.applyColorMap((labels * (255 // max(1, num_labels-1))).astype(np.uint8), cv.COLORMAP_JET)
show("10 - Labels visualization", labels_vis)

for label_id in range(num_labels):

    if label_id == background_label:
        continue

    area = int(stats[label_id, 4])
    if area < MIN_AREA or area > MAX_AREA:
        continue

    cx, cy = centroids[label_id]
    cx_i, cy_i = int(round(cx)), int(round(cy))
    kept_centers.append((cx_i, cy_i))

    # Draw detected hole center on original image
    cv.circle(img_draw, (cx_i, cy_i), 12, (0, 0, 255), -1)
    cv.putText(img_draw, f"hole", (cx_i + 10, cy_i - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

print("Detected hole centers:", kept_centers)

# ------------------------------------------------------------
# 10) Draw distances between every pair of centers
# ------------------------------------------------------------
n = len(kept_centers)

for i in range(n):
    x1, y1 = kept_centers[i]

    for j in range(i + 1, n):
        x2, y2 = kept_centers[j]

        # Draw line between centers
        cv.line(img_draw, (x1, y1), (x2, y2), (255, 0, 255), 10)

        # Euclidean distance in pixels
        d = float(np.hypot(x2 - x1, y2 - y1))

        # Put text at midpoint
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        cv.putText(img_draw, f"{d:.1f}px", (mx + 5, my + 5),
                   cv.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 255), 10)
        

# ============================================================
# 11) Final visualization
# ============================================================
show("11 - Final (centers + distances)", img_draw)
cv.waitKey(0)
cv.destroyAllWindows()
