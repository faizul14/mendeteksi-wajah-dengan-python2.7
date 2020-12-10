import  cv2, os, numpy as np
from PIL import  Image

Dirwajah = 'datawajah'
Dirlatihan = 'latihanwajah'

def getImageLabel(path):
    imagepath = [os.path.join(path,f) for f in os.listdir(path)]
    facesampel = []
    faceids = []
    for imagepath in imagepath:
        pilimg = Image.open(imagepath).convert('L') # CONVERT KE DALAM GREY
        imgnum = np.array(pilimg,'uint8')
        faceid = int(os.path.split(imagepath)[-1].split(".")[1])
        faces = faceDetector.detectMultiScale(imgnum)
        for (x, y, w, h) in faces:
            facesampel.append(imgnum[x:y+h, x:x+w])
            faceids.append(faceid)
        return  facesampel, faceid

faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

print ("System sedang melakukan training data waja. Tunggu dalam beberapa detik..")
faces, id = getImageLabel(Dirwajah)
faceRecognizer.train(faces, np.array(id))
#simpan
faceRecognizer.write(Dirlatihan+"/training.xml")
print ("sebanyak {0} data wajah telah selesai di trainingkan ke mesin..",format(len(np.unique(id))))
