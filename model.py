#importing modules
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as nump
import csv
import cv2
import os

train = ImageDataGenerator(rescale= 1/255)
check =ImageDataGenerator(rescale= 1/255)

train_dataset = train.flow_from_directory("train",
                                          target_size=(100,100),
                                          batch_size = 4,
                                          class_mode= 'binary')

check_dataset = train.flow_from_directory("check",
                                               target_size=(100,100),
                                               batch_size = 4,
                                               class_mode= 'binary')


model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation = 'relu', input_shape=(100,100,3)),
                                   tf.keras.layers.MaxPool2D(2,2),
                                    # Second convolution layer and pooling
                                   tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
                                   tf.keras.layers.MaxPool2D(2,2),
                                     # Third convolution layer and pooling
                                   tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
                                   tf.keras.layers.MaxPool2D(2,2),
                                    # Flattening the layers
                                   tf.keras.layers.Flatten(),
                                      # Adding a fully connected layer
                                   tf.keras.layers.Dense(512,activation= 'relu'),
                                   tf.keras.layers.Dense(1, activation='sigmoid')
                                  ])
# Compiling the CNN
model.compile(loss= 'binary_crossentropy',
              optimizer = Adam(learning_rate=0.001),
              metrics =['accuracy'])
#Test_predicitions
model_fit = model.fit(train_dataset,
                      epochs = 5,            
                      validation_data = check_dataset)

dir_path = 'test'


f = open("submission_1.csv",'w')
w=csv.writer(f,delimiter=',',lineterminator='\n')
w.writerow(["id","Aspectofhand"])
print(len(os.listdir(dir_path )))
for i in os.listdir(dir_path ) :
    img= image.load_img(dir_path+'//'+ i, target_size=(100,100))

    X = image.img_to_array(img)
    X = nump.expand_dims (X,axis= 0 )
    images = nump.vstack([X])
    val=model.predict(images)
    if val == 0:
        rec=[i,0]
        w.writerow(rec)
    else:
        rec=[i,1]
        w.writerow(rec)
