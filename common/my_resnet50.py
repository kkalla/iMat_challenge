# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 20:23:19 2018

@author: user
"""

from keras.applications.resnet50 import ResNet50

num_classes = 128

def load_resnet50(include_top=True,weights=None):
    if include_top:
        base_model = ResNet50(include_top=include_top,weights=weights,classes=num_classes)
        return base_model
    else:
        base_model = ResNet50(include_top=False)
    
    return base_model
    