#install modull cv2
#pip install opencv-contrib-python
#pip install pillow
#file pengenalan wajah
#detect wajah https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
#langkah untuk face recognition : rekam data wajah , trainig data wajah, recognition
import cv2, os
wajahDir = 'datawajah'
cam = cv2.VideoCapture(0)
cam.set(3, 640)#ubah lebar camera
cam.set(4, 480)#ubah tinggi camera

faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceId = input("MASUKKAN FACE ID YANG AKAN DI REKAM DATANYA [Kemudian tekan ENTER]: ")
print("Tatap wajah anda ke dalam webcam. Tunggu proses pengambilan data wajah anda..")
ambildata = 1
while True:
    retV, frame = cam.read()
    abuabu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(abuabu, 1.3, 5) #frame, factor_scala
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x,y),(x+w, y+h),(0,0,255),2)
        namafile = "wajah."+str(faceId)+"."+str(ambildata)+".jpg"
        cv2.imwrite(wajahDir+'/'+namafile, frame)
        ambildata += 1
    cv2.imshow("deteksi wajah", frame)
    #cv2.imshow("webcam 2", abuabu)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break
    elif ambildata>30:
        break
print("pengambilan data succes...")
cam.release()
cv2.destroyAllWindows()

