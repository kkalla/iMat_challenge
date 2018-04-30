# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 22:11:50 2018

@author: user
"""

import os
import keras

import pandas as pd

from utils.generators import MyGenerators

test_dir = 'data/test_images'
model_dir = 'keras_model/my_resnet50'

ids=[]
def _get_nb_train(train_dir):
    file_total = 0
    for item in os.scandir(train_dir):
        if item.is_dir():
            file_total += _get_nb_train(item.path)
        else:
            file_total += 1
            ids.append(item.name.split('.')[0])
    
    return file_total
    
    
nb_images = _get_nb_train(test_dir)
batch_size = 32
epochs=1
steps_per_epoch= nb_images/batch_size
num_classes=128

def main():
    model_path = os.path.join(model_dir,'my_resnet50.h5')
    my_resnet50 = keras.models.load_model(model_path)
    my_resnet50.summary()
    test_generator = MyGenerators().get_generator(batch_size=batch_size,shuffle=False)
    predictions = my_resnet50.predict_generator(generator=test_generator)
    print(predictions)
    results = pd.DataFrame({'id':ids,'predicted':predictions})
    print('Saving submission File...')
    results.to_csv('my_resnet50_submission.csv')
    
if __name__=="__main__":
    main()