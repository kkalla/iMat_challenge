# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 22:04:38 2018

@author: user
"""

import os
import keras

from utils.generators import MyGenerators

eval_dir = 'data/valid_images'
model_dir = 'keras_model/my_resnet50'


def _get_nb_train(train_dir):
    file_total = 0
    for item in os.scandir(train_dir):
        if item.is_dir():
            file_total += _get_nb_train(item.path)
        else:
            file_total += 1
    
    return file_total
    
    
nb_images = _get_nb_train(eval_dir)
batch_size = 32
epochs=1
steps_per_epoch= nb_images/batch_size
num_classes=128

def main():
    model_path = os.path.join(model_dir,'my_resnet50.h5')
    my_resnet50 = keras.models.load_model(model_path)
    my_resnet50.summary()
    eval_generator = MyGenerators().get_generator(batch_size=batch_size,shuffle=False)
    eval_result = my_resnet50.evaluate_generator(generator=eval_generator)
    print(eval_result)
    
if __name__=="__main__":
    main()