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
from keras.preprocessing.image import ImageDataGenerator

model_dir = 'keras_model/my_vgg19'
hparams = {'loss':'categorical_crossentropy',
           'optimizer':'adam',
           }
train_dir = 'data/train_images'

def _get_nb_train(train_dir):
    file_total = 0
    for item in os.scandir(train_dir):
        if item.is_dir():
            file_total += _get_nb_train(item.path)
        else:
            file_total += 1
    
    return file_total
    
    
nb_images = _get_nb_train(train_dir)
batch_size = 32
epochs=10
steps_per_epoch= nb_images/batch_size
num_classes=128

def main():
    print(nb_images)
#    def train_input_fn():
#        def parser(filename,label):
#            image_string = tf.read_file(filename)
#            image_decoded = tf.image.decode_jpeg(image_string)
#            image_resized = tf.image.resize_images(image_decoded,[224,224])
#            image_resized.set_shape([224,224,3])
#            label = tf.one_hot(label,depth=num_classes)
#            return {'input_1':image_resized}, label
#        filenames, labels = Data_loader().load_image_data('data/train_images')
#        filenames = tf.constant(filenames)
#        labels = tf.constant(labels)
#        
#        dataset = tf.data.Dataset.from_tensor_slices((filenames,labels))
#        dataset = dataset.map(parser)
#        dataset = dataset.shuffle(buffer_size=10000)
#        dataset = dataset.batch(batch_size)
#        iterator = dataset.make_one_shot_iterator()
#        batch_features, batch_labels = K.get_session().run(iterator.get_next())
#        while True:
#            yield batch_features,batch_labels
#    
    train_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory(
            directory=train_dir,
            target_size=(224,224),
            batch_size=batch_size,
            class_mode='categorical')
    model_path = os.path.join(model_dir,'my_vgg19.h5')
    if os.path.exists(model_path):
        my_model = keras.models.load_model(model_path)
    else:
        my_model = load_model()
    my_model.load_weights(os.path.join(model_dir,'weights.01-4.12.hdf5'))
    my_model.compile(optimizer=hparams['optimizer'],loss=hparams['loss'],metrics=['acc'])
    my_model.summary()
    class LossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
    
    history = LossHistory()
    model_checkpointer = ModelCheckpoint(
            filepath=os.path.join(model_dir,'weights.{epoch:02d}-{loss:.2f}.hdf5'),period=0.1)
    my_model.fit_generator(generator=train_generator,workers=0,verbose=1,
                           steps_per_epoch=steps_per_epoch,epochs=epochs,
                           callbacks=[history,model_checkpointer])
    print(history.losses)
    print("Saving trained weights and model...")
    my_model.save(model_path)
    
if __name__=="__main__":
    main()
