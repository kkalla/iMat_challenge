# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 20:22:41 2018

@author: user
"""

import keras

import tensorflow as tf
import keras.backend as K

from utils.data_utils import Data_loader


model_path = 'keras_model/my_vgg19/my_vgg19.h5'
num_classes = 128
batch_size = 30


def main():
    my_vgg19 = keras.models.load_model(model_path)
    
    def eval_input_fn():
        def parser(filename,label):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string)
            image_resized = tf.image.resize_images(image_decoded,[224,224])
            image_resized.set_shape([224,224,3])
            label = tf.one_hot(label,depth=num_classes)
            return {'input_1':image_resized}, label
        filenames, labels = Data_loader().load_image_data('data/train_images')
        filenames = tf.constant(filenames)
        labels = tf.constant(labels)
        
        dataset = tf.data.Dataset.from_tensor_slices((filenames,labels))
        dataset = dataset.map(parser)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        batch_features, batch_labels = K.get_session().run(iterator.get_next())
        while True:
            yield batch_features,batch_labels
    
    eval_result = my_vgg19.evaluate_generator(generator=eval_input_fn(),)
    print(eval_result)
    
if __name__=="__main__":
    main()