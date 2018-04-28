# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 14:04:03 2018

@author: user
"""

from keras.applications.vgg19 import VGG19
from keras.model import Model
from keras.layers import Flatten,Dense,Dropout
from keras.layers.normalization import BatchNormalization

num_classes=128

def load_model():
    base_model = VGG19(include_top=False,weights='imagenet',input_shape=(224,224,3))
    x = Flatten()(base_model.output)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(input = base_model.input,output=predictions)
    
    return model

