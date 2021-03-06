# -*- coding: utf-8 -*-
"""
test simple cnn model function

Created on Tue Apr 17 16:06:19 2018

@author: user
"""

import numpy as np
import tensorflow as tf

from common import simple_cnn_model

tf.logging.set_verbosity(tf.logging.INFO)

def create_fake_data():
    train_data = np.random.randn(10,800,800,3).astype(np.float32)
    train_labels = np.array([0,1,0,0,1,1,1,0,0,1])
    eval_data = np.random.randn(2,800,800,3).astype(np.float32)
    eval_labels = np.array([0,1])
    return train_data,train_labels,eval_data,eval_labels

def main():
    train_data,train_labels,eval_data,eval_labels = create_fake_data()
    # create the estimator
    classifier = tf.estimator.Estimator(
            model_fn=simple_cnn_model.simpleConvModel_fn,model_dir='tmp/convnet_test')
    
    # set up logging for predictions
    tensors_to_log={"probabilities":"softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=5)
    
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x":train_data},
            y=train_labels,
            batch_size=5,
            num_epochs=20,
            shuffle=True)
    classifier.train(input_fn=train_input_fn,steps=100,hooks=[logging_hook])
    
    # Eval the model
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x":eval_data},
            y = eval_labels,
            num_epochs=1,
            shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    
if __name__=="__main__":
    main()