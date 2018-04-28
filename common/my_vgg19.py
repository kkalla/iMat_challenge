# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 14:04:03 2018

@author: user
"""

from keras.applications.vgg19 import VGG19

def load_model():
    base_model = VGG19(include_top=True,weights='imagenet',classes=128)
    model = base_model
    
    return model

