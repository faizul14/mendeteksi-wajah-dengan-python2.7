#install modull cv2
#pip install opencv-contrib-python
#pip install pillow

import cv2
cam = cv2.VideoCapture(0)

while True:
    retV, frame = cam.read()
    abuabu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("webcam", frame)
    cv2.imshow("webcam 2", abuabu)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()