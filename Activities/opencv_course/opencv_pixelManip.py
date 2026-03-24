import numpy as np
import cv2 as cv

# Load the image
img = cv.imread('images/MCR2_Logo_Black.png')

# Check if the image was loaded successfully
if img is None:
    print("Failed to load image")
    exit()

# Get the height and width of the image
height, width, channels = img.shape

# Calculate the center coordinates
center_y = int(height / 2)
center_x = int(width / 2)

# Set a single pixel at the center to white
img[center_y, center_x] = [255, 255, 255]

# OR: Fill a region around the center (e.g., a white rectangle of size 100x100)
# Define the size of the square (half the length from the center)
square_size = 50

# Fill the region with white color
img[0:square_size, width-square_size:width] = [0,0,0]

cv.imshow("Display window", img)

# Wait indefinitely until any key is pressed
cv.waitKey(0)

# Close all OpenCV windows
cv.destroyAllWindows()