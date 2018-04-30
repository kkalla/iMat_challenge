# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 20:32:09 2018

@author: user
"""

import os
import keras

from utils.generators import MyGenerators
from common.my_resnet50 import load_resnet50
from keras.callbacks import ModelCheckpoint,Callback,TensorBoard

train_dir = 'data/train_images'
model_dir = 'keras_model/my_resnet50'
hparams = {'loss':'categorical_crossentropy',
           'metrics':['accuracy',keras.metrics.top_k_categorical_accuracy],
           'optimizer':'adagrad',
           'learning_rate':0.001,
           'decay':0.0
           }


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
epochs=50
steps_per_epoch= 100
num_classes=128

def _get_optimizer(hparams):
    if hparams['optimizer']=='adagrad':
        return keras.optimizers.Adagrad(lr=hparams['learning_rate'],decay=hparams['decay'])

def main():
    my_resnet50 = load_resnet50()
    train_generator = MyGenerators().get_generator(mode='train')
    valid_generator = MyGenerators().get_generator(mode='eval')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    checkpoints = os.listdir(model_dir)
    model_path = os.path.join(model_dir,'my_resnet50.h5')
    if os.path.exists(model_path):
        my_resnet50 = keras.models.load_model(model_path)
    if len(checkpoints) > 0:
        my_resnet50.load_weights(os.path.join(model_dir,checkpoints[-1]))

    optimizer = _get_optimizer(hparams)
    my_resnet50.compile(optimizer=optimizer,loss=hparams['loss'],
                        metrics=hparams['metrics'])
    my_resnet50.summary()
    
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
    
    history = LossHistory()
    model_checkpointer = ModelCheckpoint(
            filepath=os.path.join(model_dir,'weights.{epoch:02d}-{loss:.2f}.hdf5'),
            save_weights_only=True,period=1)
    tensorboard_writer = TensorBoard(log_dir='./logs/my_resnet50',write_images=True,histogram_freq=0)

    my_resnet50.fit_generator(generator=train_generator,workers=10,verbose=1,
                           steps_per_epoch=steps_per_epoch,epochs=epochs,
                           callbacks=[history,model_checkpointer,tensorboard_writer],validation_data=valid_generator)
    print(history.losses)
    print("Saving trained weights and model...")
    my_resnet50.save(model_path)

if __name__=="__main__":
    main()
