# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 20:41:00 2018

@author: user
"""

import keras
import os

import pandas as pd

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from keras.preprocessing.image import ImageDataGenerator

model_path = 'keras_model/my_vgg19/my_vgg19.h5'
test_data_dir = 'data/test_images'
batch_size = 32

def main():
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(224,224),
            batch_size=batch_size,
            shuffle=False,
            class_mode=None)
    my_vgg19 = keras.models.load_model(model_path)
    predictions = my_vgg19.predict_generator(generator=test_generator)
    image_ids = []
    for file in os.scandir(os.path.join(test_data_dir,'test')):
        if file.is_file():
            image_ids.append(file.name.split('.')[0])
    
    #Save results
    result = pd.DataFrame({'id':image_ids,'predicted':predictions})
    result.to_csv('my_vgg19_submisssion.csv')
    print(predictions)
    
if __name__=="__main__":
    main()
