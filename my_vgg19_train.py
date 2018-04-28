# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 14:26:14 2018

@author: user
"""
import keras
import os

import tensorflow as tf
import keras.backend as K

from common.my_vgg19 import load_model
from utils.data_utils import Data_loader
from keras.callbacks import ModelCheckpoint

batch_size = 30
epochs=10
steps_per_epoch= 200
num_classes=128
model_dir = 'keras_model/my_vgg19'
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
        batch_features, batch_labels = K.get_session().run(iterator.get_next())
        while True:
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
    model_checkpointer = ModelCheckpoint(
            filepath=os.path.join(model_dir,'weights.{epoch:02d}-{val_loss:.2f}.hdf5'))
    my_model.fit_generator(generator=train_input_fn(),workers=0,verbose=1,
                           steps_per_epoch=steps_per_epoch,epochs=epochs,
                           callbacks=[history,model_checkpointer])
    print(history.losses)
    print("Saving trained weights and model...")
    my_model.save(os.path.join(model_dir,'my_vgg19.h5'))
    
    
    
if __name__=="__main__":
    main()
