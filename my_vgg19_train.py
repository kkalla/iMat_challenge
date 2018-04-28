# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 14:26:14 2018

@author: user
"""
import keras

import tensorflow as tf

from common.my_vgg19 import load_model
from utils.data_utils import Data_loader

batch_size = 1000
epochs=1
steps_per_epoch=10
num_classes=128
hparams = {'loss':'categorical_crossentropy',
           'optimizer':'adam',
           }

def main():
    def train_input_fn():
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
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        batch_features, batch_labels = iterator.get_next()
        
        yield batch_features,batch_labels
    
    my_model = load_model()
    my_model.compile(optimizer=hparams['optimizer'],loss=hparams['loss'],metrics=['acc'])
    my_model.summary()
    class LossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
    
    history = LossHistory()
    my_model.fit_generator(generator=train_input_fn(),workers=0,verbose=1,
                           steps_per_epoch=steps_per_epoch,epochs=epochs,
                           callbacks=[history])
    print(history.losses)
    
    
    
if __name__=="__main__":
    main()