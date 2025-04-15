import cv2
import numpy as np

PATCH_RADIUS = 15
SEARCH_RADIUS = 45

def border_similarity(blemish, patch, border=1):
    """Calculate the border similarity."""
    blemishBoundary = blemish.copy()
    patchBoundary = patch.copy()
    # Preserve only boundary pixels
    blemishBoundary[border:-border, border:-border] = 0
    patchBoundary[border:-border, border:-border] = 0
    # Calculate sum of absolute differences
    diff = np.sum(cv2.absdiff(blemishBoundary, patchBoundary)) / patch.size
    similarity = 1 / (diff + 1)
    return similarity

def patch_smoothness(patch):
    """Calculate the patch smoothness."""
    grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    # Calculate gradients and their magnitude
    sobel_x = cv2.Sobel(grey, cv2.CV_32F, 1, 0)
    sobel_y = cv2.Sobel(grey, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    # Calculate sum of gradients magnitude
    roughness = np.sum(magnitude) / patch.size
    smoothness = 1 / (roughness + 1)
    return smoothness

def search_patches(image, center, patch_radius=PATCH_RADIUS, search_radius=SEARCH_RADIUS):
    """Search patches in the search area."""
    patches = []
    y, x = center
    for topleft_y in range(y - search_radius, y + search_radius - 2 * patch_radius):
        if topleft_y < 0:
            continue      
        for topleft_x in range(x - search_radius, x + search_radius - 2 * patch_radius):
            if topleft_x < 0:
                continue
            bottomright_y = topleft_y + 2 * patch_radius
            bottomright_x = topleft_x + 2 * patch_radius
            if bottomright_y < image.shape[0] and bottomright_x < image.shape[1]:
                patches.append(image[topleft_y:bottomright_y, topleft_x:bottomright_x])
    return patches

def find_patch(image, center, patch_radius=PATCH_RADIUS, search_radius=SEARCH_RADIUS):
    """Find the ideal patch."""
    y, x = center
    # Get blemish
    blemish = image[y - patch_radius:y + patch_radius, x - patch_radius:x + patch_radius]
    # Get all patches from search area
    patches = search_patches(image, center, patch_radius, search_radius)
    border_similarities = []
    smoothnesses = []
    # Calculate border similarities and patch smoothnesses
    for patch in patches:
        border_similarities.append(border_similarity(blemish, patch))
        smoothnesses.append(patch_smoothness(patch))
    # Convert to NumPy array
    border_similarities = np.array(border_similarities)
    smoothnesses = np.array(smoothnesses)
    # Get rank of border similarities
    border_similarities_ranks = np.argsort(np.argsort(-border_similarities))
    # Get rank of smoothnesses
    smoothnesses_ranks = np.argsort(np.argsort(-smoothnesses))
    # Calculate sum of ranks
    ranks_sum = border_similarities_ranks + smoothnesses_ranks
    # Find the patch with tre lowest rank (patch with highest border similarity and smoothness)
    patch = patches[np.argmin(ranks_sum)]
    return patch

def remove_blemish(center):
    """Remove the blemish."""
    global image, last_image
    y, x = center
    height, width = image.shape[:2]
    if y - PATCH_RADIUS < 0 or x - PATCH_RADIUS < 0 or y + PATCH_RADIUS > height or x + PATCH_RADIUS > width:
        print("Error: Too close to image border.")
        return
    last_image = image.copy()
    # Find the ideal patch
    patch = find_patch(image, center, patch_radius=PATCH_RADIUS, search_radius=SEARCH_RADIUS)
    # Seamlessly clone patch into image
    image = cv2.seamlessClone(patch, image, 255 * np.ones_like(patch), (x, y), cv2.NORMAL_CLONE)

def undo():
    """Undo one step."""
    global image, last_image
    image = last_image.copy()

def onMouse(action, x, y, flags, userdata):
    if action == cv2.EVENT_LBUTTONDOWN:
        remove_blemish((y, x))
    elif action == cv2.EVENT_RBUTTONDOWN:
        undo()

image = cv2.imread("./blemish.png", cv2.IMREAD_COLOR)
last_image = image.copy()

cv2.namedWindow("Blemish Remover")
cv2.setMouseCallback("Blemish Remover", onMouse)

print("Click on the blemish to remove it.")
print("Press the right mouse button to undo one step.")
print()
print(f"Patch radius: {PATCH_RADIUS}")
print(f"Search area radius: {SEARCH_RADIUS}")
print()

k = 0
while k!=27 :
  cv2.imshow("Blemish Remover", image)
  k = cv2.waitKey(20) & 0xFF

cv2.destroyAllWindows()
