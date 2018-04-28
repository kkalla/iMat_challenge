# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 20:41:00 2018

@author: user
"""

import keras
import os

import numpy as np
import pandas as pd

from PIL import Image

model_path = 'keras_model/my_vgg19/my_vgg19.h5'
test_data_dir = 'data/test_images'

def main():
    image_ids = []
    images = []
    for file in os.scandir(test_data_dir):
        _image_id = file.name.split('.')[0]    
        image_ids.append(_image_id)
        image = Image.open(file.path)
        image_resized = image.resize((224,224))
        image_list = list(image_resized.getdata())
        image_array = np.array(image_list,dtype='uint8')
        image_array = image_array.reshape((224,224,3))
        images.append(image_array)
    
    num_images = len(os.listdir(test_data_dir))
    images = np.array(images).reshape((num_images,224,224,3))
    
    my_vgg19 = keras.models.load_model(model_path)
    predictions = my_vgg19.predict(x=images)
    
    #Save results
    result = pd.DataFrame({'id':image_ids,'predicted':predictions})
    result.to_csv('my_vgg19_submisssion.csv')
    
if __name__=="__main__":
    main()