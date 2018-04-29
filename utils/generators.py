# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 20:35:13 2018

@author: user
"""

from keras.preprocessing.image import ImageDataGenerator

class MyGenerators:
    def __init__(self,
                 rescale=1./255,
                 train_data_dir='data/train_iamges',
                 eval_data_dir='data/valid_images',
                 test_data_dir='data/test_images',
                 ):
        self.rescale=rescale
        self.train_data_dir=train_data_dir
        self.eval_data_dir=eval_data_dir
        self.test_data_dir=test_data_dir
    
    def get_generator(self,batch_size=32,
                      target_size=(224,224),
                      class_mode='categorical',
                      shuffle = True):
        datagen = ImageDataGenerator(rescale=self.rescale)
        generator=datagen.flow_from_directory(
                directory=self.train_data_dir,
                target_size=target_size,
                class_mode=class_mode,
                batch_size=batch_size,
                shuffle=shuffle)
        return generator
    
        