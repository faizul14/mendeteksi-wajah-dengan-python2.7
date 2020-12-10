import cv2, os, numpy as np

wajahDir = 'datawajah'
Dirlatihan = 'latihanwajah'
cam = cv2.VideoCapture(0)
cam.set(3, 640)#ubah lebar camera
cam.set(4, 480)#ubah tinggi camera

faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()

faceRecognizer.read(Dirlatihan+"/training.xml")
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
names = ['Tidak diketahui','Fezol','Nama lain']


minwidth = 0.1*cam.get(3)
minheight = 0.1*cam.get(4)

while True:
    retV, frame = cam.read()
    frame = cv2.flip(frame, 1) #vertikal
    abuabu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(abuabu, 1.2, 5,minSize=(round(minwidth),round(minheight)),) #frame, factor_scala, minneighbor
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x,y),(x+w, y+h),(0,255,0),2)
        id, confidence = faceRecognizer.predict(abuabu[y:y+h, x:x+w]) #condidence artinya 0 berarti cocok atau sempurna
        if confidence<=50:
            nameid = names[id]
            confidenceTxt = " {0}%".format(round(100-confidence))
        else:
            nameid = names[0]
            confidenceTxt = " {0}%".format(round(100 - confidence))
        cv2.putText(frame,str(nameid),(x+5,y-5),font,1,(0,0,255),2)
        cv2.putText(frame, str(confidenceTxt), (x + 5, y+h- 5), font, 1, (255, 255, 0), 1)


    cv2.imshow("Memindai wajah", frame)
    #cv2.imshow("webcam 2", abuabu)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break
print("exit")
cam.release()
cv2.destroyAllWindows()
