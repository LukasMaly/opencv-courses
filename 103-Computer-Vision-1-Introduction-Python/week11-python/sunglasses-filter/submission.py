import cv2
import numpy as np

# Load face detector model
modelFile = 'opencv_face_detector_uint8.pb'
configFile = 'opencv_face_detector.pbtxt'
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

def detectFaceOpenCVDnn(net, image, conf_threshold=0.9):
    imageOpencvDnn = image.copy()
    imageHeight = imageOpencvDnn.shape[0]
    imageWidth = imageOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(imageOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2])[:1]:
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * imageWidth)
            y1 = int(detections[0, 0, i, 4] * imageHeight)
            x2 = int(detections[0, 0, i, 5] * imageWidth)
            y2 = int(detections[0, 0, i, 6] * imageHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(imageOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(imageHeight/150)), 8)
    return imageOpencvDnn, bboxes

# Placement of top of glasses on face
TOP = 0.3

# Glass color
GLASS_COLOR_LOWER = np.array([126, 71, 0]) / 255
GLASS_COLOR_UPPER = np.array([255, 157, 92]) / 255

# Glass opacity
OPACITY = 0.7

# Load image of glasses
glasses = cv2.imread('sunglass.png', cv2.IMREAD_UNCHANGED)

# Crop glasses
glasses = glasses[35:257, 10:613, :]

# Capture webcam
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        # Detect face
        output, bboxes = detectFaceOpenCVDnn(net, frame)
        
        if len(bboxes) > 0:
            x1, y1, x2, y2 = bboxes[0]
            if x1 > 0 and y1 > 0:
                face = frame[y1:y2, x1:x2, :].copy()
                face_height, face_width = face.shape[:2]
                
                # Resize glasses
                glasses_width = face_width
                glasses_height = int((glasses_width / glasses.shape[1]) * glasses.shape[0])
                glasses_resized = cv2.resize(glasses, (glasses_width, glasses_height))

                # Convert to float
                face = np.float32(face) / 255
                glasses_resized = np.float32(glasses_resized) / 255

                # Separate glass channels
                glassesBGR = glasses_resized[:, :, 0:3]
                glassesMask = glasses_resized[:, :, 3]
                
                # Get glass mask
                glassMask = cv2.inRange(glassesBGR, GLASS_COLOR_LOWER, GLASS_COLOR_UPPER) / 255
                
                # Make glass opaque
                glassesMask = glassesMask - (1 - OPACITY) * glassMask
                
                # Merge glass mask
                glassesMask = cv2.merge((glassesMask, glassesMask, glassesMask))
                
                # Place glasses on face
                top = int(TOP * face_height)
                face[top:(top + glasses_height), :] = face[top:(top + glasses_height), :] * (1 - glassesMask)
                face[top:(top + glasses_height), :] = face[top:(top + glasses_height), :] + glassesBGR * glassesMask
                
                # Place face with glassses into original image
                frame[y1:y2, x1:x2] = np.uint8(face * 255)

        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) == 27:
            break

    else: 
        break

cap.release()
