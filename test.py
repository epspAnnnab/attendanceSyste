import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pickle

path ='ImagesModels'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def finencodings(images):
    encodeList = []
    for img in images:
        img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode =face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name,img):
    with open('attendance.csv','r+') as f :
        find = 0
        myDataList =f.readlines()
        nameList=[]
        for line in myDataList:
            entry =line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            cv2.imwrite('ImageSaved/'+name+'.jpg', img)
          
           

# encodeListKnown = finencodings(images)


encodeListKnown = []
# with open('encodeFile.dat','wb') as f:
#     pickle.dump(encodeListKnown,f)
with open('encodeFile.dat','wb') as f:
    pickle.dump(encodeListKnown,f)

   
   
print('end')

cap = cv2.VideoCapture(0)

while True:
    success , img = cap.read()
    imgs = cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgs)
    encodeCurFrame = face_recognition.face_encodings(imgs,faceCurFrame)
    
    for encodeFace,faceloc in zip(encodeCurFrame,faceCurFrame):
        matches =face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        matchIndex =np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1,x2,y2,x1=faceloc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255))
            
            markAttendance(name.upper(),img)
        else:
            y1,x2,y2,x1=faceloc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
            cv2.putText(img,'inconnu',(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255))


    cv2.imshow('webcam',img)
    cv2.waitKey(1)

