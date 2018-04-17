# -*- coding: utf-8 -*-
"""
Simple convolutional network model

Created on Tue Apr 17 14:25:57 2018

@author: user
"""

import tensorflow as tf

def simpleConvModel_fn(features, labels, mode):
    """Simple CNN model function
    CONV->(ReLU) -> POOL -> CONV->(ReLU) -> POOL -> DENSE -> DENSE
    
    """
    # Input Layer
    input_layer = tf.reshape(features["x"],[-1,800,800,3])
    
    # Conv layer #1
    conv1 = tf.layers.conv2d(inputs = input_layer,
                             filters = 32,
                             kernel_size = [11,11],
                             strides = 3,
                             padding = "valid",
                             activation = tf.nn.relu)
    # Pooling layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2],strides=2)
    # Conv layer #2
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[9,9],
            strides=3,
            padding="valid",
            activation=tf.nn.relu)
    # Pooling layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2)
    # Conv layer #3
    conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=64,
            kernel_size=[7,7],
            strides=2,
            padding="valid",
            activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3,pool_size=[2,2],strides=2)
    
    #Dense layer
    pool3_flat = tf.reshape(pool3,[-1,4*4*64])
    dense = tf.layers.dense(inputs=pool3_flat,units=1024,activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense,rate=0.4,
                                training= mode == tf.estimator.ModeKeys.TRAIN )
    
    #Logits layer
    logits = tf.layers.dense(inputs=dropout,units=128)
    predictions = {"classes": tf.argmax(input=logits, axis=1),
                   "probabilities": tf.nn.softmax(logits,name="softmax_tensor")}
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)
    
    if mode==tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)
    
    eval_metric_ops={
            "accuracy":tf.metrics.accuracy(labels=labels,
                                           predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)
    