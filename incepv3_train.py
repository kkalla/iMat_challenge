# -*- coding: utf-8 -*-
"""
Inceptionv3 model of keras applications(pre-trained using imageNet data)

Created on Fri Apr 27 13:30:29 2018

@author: user
"""
import os

import tensorflow as tf

from utils.data_utils import Data_loader
from tf.keras.applications.inception_v3 import InceptionV3

batch_size = 30
model_dir = os.path.join(os.getcwd(),'log/inception_v3')
max_train_step = 500
hparams = {
        'optimizer':'Adagrad',
        'loss':'categorical_crossentropy'
        }

def main():
    incepv3_model = InceptionV3(include_top=False, weights='imagenet',
                                input_shape=(800,800,3))
    incepv3_model.compile(optimizer=hparams['optimizer'],loss=hparams['loss'],
                          metrics=['accuracy'],)
    
    incepv3_classifier = tf.keras.estimator.model_to_estimator(
            keras_model=incepv3_model,model_dir=model_dir)
    
    def train_input_fn():
        def parser(filename,label):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string)
            image_resized = tf.image.resize_images(image_decoded,[800,800])
            label = tf.one_hot(label,depth=128)
            return {'features':image_resized}, label
        filenames, labels = Data_loader().load_image_data('data/train_images')
        filenames = tf.constant(filenames)
        labels = tf.constant(labels)
        
        dataset = tf.data.Dataset.from_tensor_slices((filenames,labels))
        dataset = dataset.map(parser)
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        batch_features, batch_labels = iterator.get_next()
        
        return batch_features,batch_labels
    
    incepv3_classifier.train(input_fn=train_input_fn(),max_steps=max_train_step)
    
    
    
if __name__=="__main__":
    main()