import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pickle
import pyttsx3
from playsound import playsound

path ='ImagesModels'
classNames = []
encodeList = []
encode = {}
myList = os.listdir(path)
print(myList)
print(classNames)

#engine = pyttsx3.init()



def saveEncoding():
    for cl in myList:
     curImg =  cv2.imread(f'{path}/{cl}')
     classNames.append(os.path.splitext(cl)[0])
     curImg = cv2.cvtColor(curImg,cv2.COLOR_BGR2RGB)
     encode [""+os.path.splitext(cl)[0]] =  face_recognition.face_encodings(curImg)[0]
    with open('encodeFile1.dat','wb') as f:
       pickle.dump(encode,f)
   




def markAttendance(name,img):
    with open('attendance.csv','r+') as f : 
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
            playsound('pip.wav')
            #engine.say(name+" OK")
           # engine.runAndWait()
          
           

#saveEncoding()c

with open('encodeFile1.dat','rb') as f:
    all_face_encoding =pickle.load(f)

classNames = list(all_face_encoding.keys())
encodeListKnown = np.array(list(all_face_encoding.values()))
   
   
print('Start ...')

cap = cv2.VideoCapture(0)

while True:
    success , img = cap.read()
    imgs = cv2.resize(img,(0,0),None,0.5,0.5)
    imgs = cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgs)
    encodeCurFrame = face_recognition.face_encodings(imgs,faceCurFrame)
    
    for encodeFace,faceloc in zip(encodeCurFrame,faceCurFrame):
        matches =face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        matchIndex =np.argmin(faceDis)
        print(faceDis[matchIndex])

        if faceDis[matchIndex] < 0.39:
            name = classNames[matchIndex].upper()
            y1,x2,y2,x1=faceloc
            y1,x2,y2,x1 = y1*2,x2*2,y2*2,x1*2
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0))
            cv2.putText(img,'EPSP ANNABA V 1.1',(0,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
            
            markAttendance(name.upper(),img)
        else:
            y1,x2,y2,x1=faceloc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
            cv2.putText(img,'inconnu',(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255))


    #cv2.imshow('webcam',img)
    # cv2.namedWindow("foo", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("foo", cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
    cv2.imshow("foo", img)
    cv2.waitKey(200)

