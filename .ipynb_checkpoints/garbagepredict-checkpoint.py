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


model = tf.keras.models.load_model("garbage.model")
CATEGORIES = ['cardboard','glass','metal','paper','plastic','trash']

def predict():
    predict_datagen = ImageDataGenerator(validation_split=0.1,rescale=1./255)

    img1 = cv2.imread('gs1.jpg')
    img1 = cv2.resize(img1, (300, 300))
    img1 = img1.reshape(1, 300, 300, -1)
    print(img1.shape)

    prediction = model.predict(img1)
    print(prediction)
    n=0
    for i in prediction[0]:
        if i==1.0:
            print(CATEGORIES[n])
            map(n)
        n=n+1

def map(n):
    arduino = serial.Serial('COM3', 9600)                                         #Creates arduino object and establishes connection to port (Enter your port)
    time.sleep(2)                                                                               #waits for connection to establish                                                                      #prints input
    if(n == '0'):                                                                                #if input is 1
        arduino.write(b'0')                                                                          #sends '1' to arduino
        print("Cardboard")
        time.sleep(1)
    if(n == '1'):                                                                                #if input is 1
        arduino.write(b'1')                                                                          #sends '1' to arduino
        print("Glass")
        time.sleep(1)
    if(n == '2'):                                                                                #if input is 1
        arduino.write(b'2')                                                                          #sends '1' to arduino
        print("Metal")
        time.sleep(1)
    if(n == '3'):                                                                                #if input is 1
        arduino.write(b'3')                                                                          #sends '1' to arduino
        print("Paper")
        time.sleep(1)
    if(n == '4'):                                                                                #if input is 1
        arduino.write(b'4')                                                                          #sends '1' to arduino
        print("Plastic")
        time.sleep(1)
    if(n == '5'):                                                                                #if input is 1
        arduino.write(b'5')                                                                          #sends '1' to arduino
        print("Trash")





