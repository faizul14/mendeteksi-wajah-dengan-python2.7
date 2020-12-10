#install modull cv2
#pip install opencv-contrib-python
#pip install pillow
#file pengenalan wajah
#detect wajah https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
#detect mata https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml


import cv2
cam = cv2.VideoCapture(0)
cam.set(3, 640)#ubah lebar camera
cam.set(4, 480)#ubah tinggi camera

faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mataDetector2 = cv2.CascadeClassifier('haarcascade_eye.xml')


while True:
    retV, frame = cam.read()
    abuabu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(abuabu, 1.3, 5) #frame, factor_scala
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x,y),(x+w, y+h),(0,0,255),2)
        faezolabuabu = abuabu[y:y+h, x:x+w]
        faezolwarna = frame[y:y+h, x:x+w]
        mata = mataDetector2.detectMultiScale(faezolabuabu)
        for (xe, ye, we, he) in mata:
            cv2.rectangle(faezolwarna, (xe,ye),(xe+we, ye+he), (0,0,255),1)

    cv2.imshow("deteksi wajah", frame)
    #cv2.imshow("webcam 2", abuabu)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

