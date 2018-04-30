# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 20:35:13 2018

@author: user
"""

from keras.preprocessing.image import ImageDataGenerator

class MyGenerators:
    def __init__(self,
                 rescale=1./255,
                 train_data_dir='data/train_images',
                 eval_data_dir='data/valid_images',
                 test_data_dir='data/test_images',
                 ):
        self.rescale=rescale
        self.train_data_dir=train_data_dir
        self.eval_data_dir=eval_data_dir
        self.test_data_dir=test_data_dir
    
    def get_generator(self,mode,batch_size=32,
                      target_size=(224,224),
                      class_mode='categorical',
                      shuffle = True):
        datagen = ImageDataGenerator(rescale=self.rescale)
        if mode=='train':
            directory = self.train_data_dir
        elif mode=='eval':
            directory = self.eval_data_dir
        else:
            directory = self.test_data_dir
        generator=datagen.flow_from_directory(
                directory=directory,
                target_size=target_size,
                class_mode=class_mode,
                batch_size=batch_size,
                shuffle=shuffle)
        return generator
    
        
