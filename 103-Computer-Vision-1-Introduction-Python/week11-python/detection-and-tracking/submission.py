import cv2
import numpy as np


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 2)


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs, targetClasses):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            if detection[4] > objectnessThreshold :
                scores = detection[5:]
                classId = np.argmax(scores)
                if classes[classId] in targetClasses:
                    confidence = scores[classId]
                    if confidence > confThreshold:
                        center_x = int(detection[0] * frameWidth)
                        center_y = int(detection[1] * frameHeight)
                        width = int(detection[2] * frameWidth)
                        height = int(detection[3] * frameHeight)
                        left = int(center_x - width / 2)
                        top = int(center_y - height / 2)
                        classIds.append(classId)
                        confidences.append(float(confidence))
                        boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    nmsboxes = []
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        nmsboxes.append(box)
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

    return nmsboxes


def detect(targetClasses):
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    # Sets the input to the network
    net.setInput(blob)
    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))
    # Remove the bounding boxes with low confidence
    boxes = postprocess(frame, outs, targetClasses)

    return boxes


def track(frame, bbox=None):
    global tracker
    # Update tracker
    if bbox == None:
        ok, bbox = tracker.update(frame)
        if ok:
            # Display bounding box.
            pt1 = (int(bbox[0]), int(bbox[1]))
            pt2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
    # Initialize tracker
    else:
        tracker = cv2.TrackerKCF_create()
        ok = tracker.init(frame, tuple(bbox))
    return ok


# Initialize the parameters
objectnessThreshold = 0.5 # Objectness threshold
confThreshold = 0.5       # Confidence threshold
nmsThreshold = 0.4        # Non-maximum suppression threshold
inpWidth = 416            # Width of network's input image
inpHeight = 416           # Height of network's input image

# Load names of classes
classesFile = 'coco.names'
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

# Load the model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

cap = cv2.VideoCapture('soccer-ball.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)                        # frame rate of video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # width of video
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # height of video
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width, frame_height))

tracker = None
tracking = False

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        if not tracking:
            boxes = detect(['sports ball'])
            if len(boxes) > 0:
                cv2.putText(frame, "Detecting", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 178, 50), 2)
                tracking = track(frame, boxes[0])
            else:
                cv2.putText(frame, "Not detecting or tracking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            tracking = track(frame)
            if tracking:
                cv2.putText(frame, "Tracking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Not detecting or tracking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out.write(frame)
        cv2.imshow("Video Output", frame)
        cv2.waitKey(1)
    else: 
        break

cap.release()
out.release()
