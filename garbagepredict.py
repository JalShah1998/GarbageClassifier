#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Flatten, MaxPooling2D,Dense,Dropout
from keras.models  import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
import random,os,glob
import matplotlib.pyplot as plt
import serial
import time


# In[2]:


model = tf.keras.models.load_model("trained_model.h5")


# In[3]:


CATEGORIES = ['cardboard','glass','metal','paper','plastic','trash']


# In[4]:


def predict(path):
    predict_datagen = ImageDataGenerator(validation_split=0.1,rescale=1./255)

    img1 = cv2.imread(path)
    img1 = cv2.resize(img1, (300, 300))
    img1 = img1.reshape(1, 300, 300, -1)
    print(img1.shape)

    prediction = model.predict(img1)
    print(prediction)
    n=0
    for i in prediction[0]:
        if i==1.0:
            print(CATEGORIES[n])
            mapr(n)
        n=n+1


# In[ ]:





# In[5]:


def mapr(n):
    print("entered mapr",n)
    arduino = serial.Serial('COM3', 9600)                                         #Creates arduino object and establishes connection to port (Enter your port)
    time.sleep(2)                                                                               #waits for connection to establish                                                                      #prints input
    if(n == 0):                                                                                #if input is 1
        arduino.write(b'0')                                                                          #sends '1' to arduino
        print("Cardboard")
        time.sleep(1)
    if(n == 1):                                                                                #if input is 1
        arduino.write(b'1')                                                                          #sends '1' to arduino
        print("Glass")
        time.sleep(1)
    if(n == 2):                                                                                #if input is 1
        arduino.write(b'2')                                                                          #sends '1' to arduino
        print("Metal")
        time.sleep(1)
    if(n == 3):                                                                                #if input is 1
        arduino.write(b'3')                                                                          #sends '1' to arduino
        print("Paper")
        time.sleep(1)
    if(n == 4):                                                                                #if input is 1
        arduino.write(b'4')                                                                          #sends '1' to arduino
        print("Plastic")
        time.sleep(1)
    if(n == 5):                                                                                #if input is 1
        arduino.write(b'5')                                                                          #sends '1' to arduino
        print("Trash")


# In[ ]:





# In[6]:


key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)
def saveimage():
            cv2.imwrite(filename='saved_img.jpg', img=frame)
            webcam.release()
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
            print("Image saved!")
while True:
    try:
        
        check, frame = webcam.read()
        print(check) #prints true as long as the webcam is running
        print(frame) #prints matrix values of each framecd
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        saveimage()
        break
        if key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
        
    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break


# In[7]:


path1='saved_img.jpg'
predict(path1)


# In[ ]:





# In[ ]:




