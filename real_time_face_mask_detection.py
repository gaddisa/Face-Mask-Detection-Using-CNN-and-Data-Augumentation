# -*- coding: utf-8 -*-
"""
Created on Wed May 25 06:58:10 2022

@author: gaddisa
"""


from keras.models import load_model
import numpy as np
import cv2

model = load_model('New_Face_Mask_Model.h5')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

vid_source = cv2.VideoCapture(0)

#Dictionaries containing detail of wearing mask as well as putting rectangle around face
text_dict = {0:"Mask ON",1: "No Mask"}
rect_color = {0:(0,255,0),1:(0,0,255)}

Subject = "Breach of Vistor Policy Detected"
Text =  "Vistor without face mask detected. Check camerato recognize vistor"

while(True):
    ret,img = vid_source.read()
    img = np.array(img[:,::-1]) # cast as array
    grayscale_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale_img,1.3,5)
    for (x,y,w,h) in faces:
        face_img = grayscale_img[y:y+w,x:x+w]
        resized_img = cv2.resize(face_img,(112,112))
        normal_png = resized_img/255.0
        reshaped_img = np.reshape(normal_png,(1,112,112,1))
        result = model.predict(reshaped_img)
        label = np.argmax(result,axis = 1)[0]
        cv2.rectangle(img,(x,y-40),(x+w,y),rect_color[label],-1)

        cv2.rectangle(img,(x,y),(x+w,y+h),rect_color[label],2)
        cv2.putText(img,text_dict[label],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)
        
    cv2.imshow('Live Feed',img)
    key = cv2.waitKey(1)
    
    if (key == 27):
        break
cv2.destroyAllWindows()
vid_source.release()