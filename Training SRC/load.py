# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 23:46:22 2020

@author: SURYA
"""
import tensorflow as tf
import numpy as np

def init():
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
    model.load_weights("weights.h5")
    
    print("Model loaded!!")
    
    model.compile(optimizer='adam',
              loss= "categorical_crossentropy",
              metrics=['accuracy'])
    graph = tf.get_default_graph()

    return model, graph