# -*- coding: utf-8 -*-
"""
TRAIN|VALID|TEST simple model

Created on Tue Apr 17 21:45:21 2018

@author: user
"""
from __future__ import absolute_import
import os

import numpy as np
import tensorflow as tf

from common import simple_cnn_model
from utils.data_utils import Data_loader

tf.logging.set_verbosity(tf.logging.INFO)

def create_fake_data():
    train_data = np.random.randn(10,800,800,3).astype(np.float32)
    train_labels = np.array([0,1,0,0,1,1,1,0,0,1])
    eval_data = np.random.randn(2,800,800,3).astype(np.float32)
    eval_labels = np.array([0,1])
    return train_data,train_labels,eval_data,eval_labels

def main():
    #Loading datasets
    train_data,train_labels,eval_data,eval_labels=create_fake_data()
    #Create estimator
    classifier = tf.estimator.Estimator(
            model_fn=simple_cnn_model.simpleConvModel_fn,model_dir='log/simpleConv')
    #Set up logging
    tensors_to_log={'loss':'loss'}
    logging_hook=tf.train.LoggingTensorHook(
            tensors=tensors_to_log,every_n_iter=1000)
    
    #Train the model
#    train_input_fn = tf.estimator.inputs.numpy_input_fn(
#            x={'x':train_data},
#            y=train_labels,
#            batch_size=10000,
#            num_epochs=1000,
#            shuffle=True)
    def train_input_fn():
        def parser(filename,label):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string)
            image_resized = tf.image.resize_images(image_decoded,[800,800])
            return {'x':image_resized}, label
        filenames, labels = Data_loader().load_image_data('data/train_images')
        filenames = tf.constant(filenames)
        labels = tf.constant(labels)
        
        dataset = tf.data.Dataset.from_tensor_slices((filenames,labels))
        dataset = dataset.map(parser)
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(30)
        dataset = dataset.repeat(10)
        iterator = dataset.make_one_shot_iterator()
        
        features, labels = iterator.get_next()
        return features, labels
        
        
#    classifier.train(input_fn=train_input_fn,hooks=[logging_hook])
    
    #Eval the model
#    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#            x={'x':eval_data},
#            y=eval_labels,
#            num_epochs=1,
#            shuffle=False)
    def eval_input_fn():
        def parser(filename,label):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string)
            image_resized = tf.image.resize_images(image_decoded,[800,800])
            return {'x':image_resized}, label
        filenames, labels = Data_loader().load_image_data('data/valid_images')
        filenames = tf.constant(filenames)
        labels = tf.constant(labels)
        
        dataset = tf.data.Dataset.from_tensor_slices((filenames,labels))
        dataset = dataset.map(parser)
        dataset = dataset.batch(32)
        iterator = dataset.make_one_shot_iterator()
        
        features, labels = iterator.get_next()
        return features, labels
   
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    
if __name__=="__main__":
    main()
