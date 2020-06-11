# -*- coding: utf-8 -*-
"""
Created on Thu May 21 23:10:45 2020

@author: SURYA
"""
#import required modules
import tensorflow as tf
import cv2
import tensorflow.keras.datasets.mnist as mnist
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import json


#To import and split the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
class_names=[0,1,2,3,4,5,6,7,8,9]


#To check the number of dataset available
print("Number of training examples: {}".format(X_train.shape[0]))
print("Number of test examples:     {}".format(X_test.shape[0]))

#resize and normalise
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train/=255
X_test/=255
#onehot encoder
number_of_classes = 10

Y_train = tf.keras.utils.to_categorical(y_train, number_of_classes)
Y_test = tf.keras.utils.to_categorical(y_test, number_of_classes)

#display 10 images
plt.figure(figsize=(10,10))
i = 0
for j in range(0,10):
    image = X_train[j].reshape((28,28))
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[j]])
    i += 1
plt.show()



#augmentation for preventing overfitting :)
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)

test_gen = ImageDataGenerator()
train_generator = gen.flow(X_train, Y_train, batch_size=64)
test_generator = test_gen.flow(X_test, Y_test, batch_size=64)
x_batch, y_batch = next(train_generator)

#plot 2 augmented photos
for i in range (0,2):
    image = x_batch[i].reshape(28,28)
    plt.imshow(image,cmap=plt.cm.binary)
    plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,
                           input_shape=(28, 28,1)),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(.25),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])


model.compile(optimizer='adam',
              loss= "categorical_crossentropy",
              metrics=['accuracy'])


model.fit(train_generator,
          epochs=3,
          steps_per_epoch=math.ceil(60000/64),
          validation_data=test_generator,
          validation_steps=10000//64)






#test with custom files
x=cv2.imread("test.png",cv2.IMREAD_GRAYSCALE)
#x=tf.cast(x, tf.float32)
x = cv2.resize(x, dsize=(28, 28), interpolation=cv2.INTER_CUBIC).astype(float)
x/=255
x = 1-x.reshape((28,28))
plt.imshow(x, cmap=plt.cm.binary)
x = x.reshape((1,28,28,1))

print(np.argmax(model.predict(x)))
print("Probablity distribution:{}".format(model.predict(x)))




with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
model.save_weights('weights.h5')
