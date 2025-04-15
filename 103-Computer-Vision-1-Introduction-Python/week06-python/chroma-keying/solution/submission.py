import cv2
import numpy as np

PATCH_RADIUS = 1

def bgr2ycrcb(bgr):
    b, g, r = bgr
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128
    cb = (b - y) * 0.564 + 128
    return (y, cr, cb)

key_bgr = np.float32((52, 166, 70))
key_ycrcb = bgr2ycrcb(key_bgr)
tolerance = 0.2

def select_color(action, x, y, flags, userdata):
    global key_bgr, key_ycrcb
    if action == cv2.EVENT_LBUTTONDOWN:
        patch = frame[y - PATCH_RADIUS:y + PATCH_RADIUS + 1, x - PATCH_RADIUS:x + PATCH_RADIUS + 1, :]
        key_bgr = np.uint8(np.mean(patch.reshape(-1, 3), axis=0))
        key_ycrcb = bgr2ycrcb(key_bgr)

def set_tolerance(value):
    global tolerance
    tolerance = value / 100

background = cv2.imread('./background.jpg')
cap = cv2.VideoCapture('./greenscreen-demo.mp4')

cv2.namedWindow("Chroma Keying", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Chroma Keying", select_color)
cv2.createTrackbar("Tolerance", "Chroma Keying", int(tolerance * 100), 100, set_tolerance)

# Check if stream opened successfully
if (cap.isOpened() == False): 
    print("Error in opening video file.")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width, frame_height))

k = 0
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True and k != 27:
        frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        distance = np.sqrt(np.power(frame_ycrcb[:, :, 1] - key_ycrcb[1], 2) + np.power(frame_ycrcb[:, :, 2] - key_ycrcb[2], 2))
        tol_inner = tolerance * 128 - tolerance * 16
        tol_outer = tolerance * 128 + tolerance * 16
        mask = (distance - tol_inner) / (tol_outer - tol_inner)
        mask[distance < tol_inner] = 0
        mask[distance > tol_outer] = 1
        mask = np.expand_dims(mask, axis=2)
        result = (frame * mask + background * (1 - mask)).astype(np.uint8)
        out.write(result)
        cv2.imshow("Chroma Keying", result)
        k = cv2.waitKey(1) & 0xFF

    # Break the loop
    else: 
        break

cap.release()
out.release()
