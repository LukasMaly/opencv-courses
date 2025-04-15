import cv2
import math
 
# Lists to store the points
topleft=[]
bottomright=[]
 
def drawSquare(action, x, y, flags, userdata):
  # Referencing global variables 
  global topleft, bottomright
  # Action to be taken when left mouse button is pressed
  if action==cv2.EVENT_LBUTTONDOWN:
    topleft=[(x,y)]
    # Mark the vertex
    cv2.circle(source, topleft[0], 1, (255,255,0), 2, cv2.LINE_AA )
 
  # Action to be taken when left mouse button is released
  elif action==cv2.EVENT_LBUTTONUP:
    bottomright=[(x,y)]
    # Mark the vertex
    cv2.circle(source, bottomright[0], 1, (255,255,0),2, cv2.LINE_AA )
    # Draw the square
    cv2.rectangle(source, topleft[0], bottomright[0], (255,255,0), 2, cv2.LINE_AA)
    cv2.imwrite("face.png",source[topleft[0][1]:bottomright[0][1],\
            topleft[0][0]:bottomright[0][0]])
    cv2.imshow("Window",source)
 
 
source = cv2.imread("../data/images/sample.jpg",1)
# Make a dummy image, will be useful to clear the drawing
dummy = source.copy()
cv2.namedWindow("Window")
# highgui function called when mouse events occur
cv2.setMouseCallback("Window", drawSquare)
k = 0
# loop until escape character is pressed
while k!=27 :
  
  cv2.imshow("Window", source)
  cv2.putText(source,'''Choose top left corner, and drag, 
                      Press ESC to exit and c to clear''' ,
              (10,30), cv2.FONT_HERSHEY_SIMPLEX, 
              0.7,(255,255,255), 2 );
  k = cv2.waitKey(20) & 0xFF
  # Another way of cloning
  if k==99:
    source= dummy.copy()
 
cv2.destroyAllWindows()
