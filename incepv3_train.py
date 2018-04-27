# -*- coding: utf-8 -*-
"""
Inceptionv3 model of keras applications(pre-trained using imageNet data)

Created on Fri Apr 27 13:30:29 2018

@author: user
"""

import tensorflow as tf

from utils.data_utils import Data_loader
from keras.layers import Input
from keras.applications.inception_v3 import InceptionV3

hparams = {
        'optimizer':'Adagrad',
        'loss':'categorical_crossentropy'
        }

def main():
    def train_input_fn():
        def parser(filename,label):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string)
            image_resized = tf.image.resize_images(image_decoded,[800,800])
            label = tf.one_hot(label,num_classes=128)
            return {'x':image_resized}, label
        filenames, labels = Data_loader().load_image_data('data/train_images')
        filenames = tf.constant(filenames)
        labels = tf.constant(labels)
        
        dataset = tf.data.Dataset.from_tensor_slices((filenames,labels))
        dataset = dataset.map(parser)
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(30)
        iterator = dataset.make_one_shot_iterator()
        
        features, labels = iterator.get_next()
        return features, labels
    
    incepv3_model = InceptionV3(include_top=False, weights='imagenet',
                                input_shape=(800,800,3))
    incepv3_model.compile(optimizer=hparams['optimizer'],loss=hparams['loss'],
                          metrics=['accuracy'],)
    incepv3_model.fit_generator(generator = train_input_fn(),
                    steps_per_epoch=200,
                    workers = 0 , # This is important
                    verbose = 1)
    
    
if __name__=="__main__":
    main()