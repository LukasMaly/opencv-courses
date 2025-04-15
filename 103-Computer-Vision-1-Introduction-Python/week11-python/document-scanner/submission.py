import cv2
import numpy as np


def order_points(pts):
    # Source: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    # return the ordered coordinates
    return rect


# Set document parameters
DOCUMENT_WIDTH = 500
APSECT_RATIO = np.sqrt(2)
DOCUMENT_HEIGHT = int(DOCUMENT_WIDTH * APSECT_RATIO)

# Read image
image = cv2.imread('scanned-form.jpg')
image_copy = image.copy()

# Detect edges
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(gray, 100, 200)

# Find contours
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get contour with largest area
cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]

# Draw contour
cv2.drawContours(image_copy, [cnt], -1, (0, 255, 0), 3)

# Approximate contour
peri = cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
points = approx.reshape(4, 2)

# Draw points
for point in points:
    cv2.circle(image_copy, tuple(point), 5, (0, 0, 255), -1)

# Order points
rect = order_points(points)

# Define points in output image
dst = np.array([
    [0, 0],
    [DOCUMENT_WIDTH - 1, 0],
    [DOCUMENT_WIDTH - 1, DOCUMENT_HEIGHT - 1],
    [0, DOCUMENT_HEIGHT - 1]], dtype = "float32")

# Get perspective transform and warp image
M = cv2.getPerspectiveTransform(rect, dst)
warped = cv2.warpPerspective(image, M, (DOCUMENT_WIDTH, DOCUMENT_HEIGHT))

# Write output image
cv2.imwrite('output.png', warped)

# Show input and output images
cv2.imshow('Input image', image_copy)
cv2.imshow('Output image', warped)
cv2.waitKey(0)
