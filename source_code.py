# -*- coding: utf-8 -*-
"""
Created on Wed May 25 06:31:23 2022

@author: gaddisa
"""

import os

#setting directory path for dataset
Dataset = "data"
Data_dir = os.listdir(Dataset)
print(Data_dir)

import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

img_rows,img_cols = 112,112

images = []
labels = [] 

for category in Data_dir:
  folder_path = os.path.join(Dataset,category)
  for img in os.listdir(folder_path):
    img_path = os.path.join(folder_path,img)
    img = cv2.imread(img_path)

    try:
      #converting image into gray scale
      grayscale_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

      #resizing gray scaled image into size 56x56
      resized_img =  cv2.resize(grayscale_img,(img_rows,img_cols))
      images.append(resized_img)
      labels.append(category)
    #Exception Handling
    except Exception as e:
      print("Exception Details : " + e)
images = np.array(images) /255.0
images = np.reshape(images,(images.shape[0],img_rows,img_cols,1))

#performing one hot encoding
lb = LabelBinarizer() 
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
labels = np.array(labels)

(train_X,test_X,train_y,test_y) = train_test_split(images,labels,test_size = 0.25,random_state= 0)


"""
Building CNN Classification Model**
"""

from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D

num_classes = 2
batch_size = 32

#Building CNN model via Sequential API

model = Sequential()

#First Layer
model.add(Conv2D(64,(3,3), input_shape = (img_rows,img_cols,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

#Second Layer
model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

#Flatten and Dropout Layer
model.add(Flatten())
model.add(Dropout(0.5))

#Softmax Classifier
model.add(Dense(64,activation = "relu"))
model.add(Dense(num_classes,activation= "softmax"))

print(model.summary())


"""
**Train the Model**"""
epochs = 50

model.compile(loss= 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
fitted_model = model.fit(train_X,train_y,epochs = epochs,validation_split=0.25)

"""
**Plot the training loss and accuracy**
"""
import matplotlib.pyplot as plt
plt.plot(fitted_model.history['loss'],'r',label = "Training  Loss")
plt.plot(fitted_model.history['val_loss'],label = "Validation Loss")
plt.xlabel("# of Epochs")
plt.ylabel("Loss Value")
plt.legend()
plt.show()

#Plotting Accuracy
plt.plot(fitted_model.history['accuracy'],'r',label = "Training Accuracy")
plt.plot(fitted_model.history['val_accuracy'],label = "Validation Accuracy")
plt.xlabel("# of Epochs")
plt.ylabel("Accuracy Value")
plt.legend()
plt.show()

"""
Save the model
"""
model.save("New_Face_Mask_Model.h5")    





