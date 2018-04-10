# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 19:05:46 2018

@author: kkalla

Building blocks for Convolutional Nueral Networks w/o TensorFlow
"""

import numpy as np

class Conv:
    """Simple Convolutional Layer.
    
    Arguments:
        num_input_channels: int, number of channels of input signal.
        filter_size: window-size of filter, default = 3
        padding: one of ['same','valid'], if padding = 'valid', then no padding.
            If padding = 'same', then output size will be equal to input image size.
        stride: int, default = 2
        num_filters: number of filters
        initializer: str,
            'xavier_normal': Initializing weights with normal distribution centered
                on 0 with stddev is sqrt(2/(input_dim + output_dim))
    
    """
    def __init__(
            self,num_input_channels, num_filters, filter_size = 3, padding='same',
            stride=2,initializer='xavier_normal'):
        
        #Initializing weights
        if initializer == 'xavier_normal':
            self.W = np.random.randn(
                    num_filters,filter_size,filter_size,num_input_channels)\
                    /np.sqrt(2/(num_filters+num_input_channels))
        else:
            print("Error: check weights initializer!!")
            return
        #Initializing bias
        self.b = np.zeros((num_filters,1,1,1))
        
        self.filter_size = filter_size
        self.padding = padding
        self.stride = stride
        self.num_filters = num_filters
        
    def forward(self,input_X):
        """Compute forward propagation
        
        Argument:
            input_X: 4-D numpy array, of shape (batch_size, height, width, #channels)
            
        Return:
            output: 4-D numpy array, 
                of shape (batch_size, output_height, output_width, num_filters)
                output_height and output_width are calculated using this formula
                output_height = (input_height+2*padding-filter_height)/2 + 1,
                output_width = (input_width+2*padding-filter_width)/2 + 1.
        """
        batch_size,in_h,in_w,num_in_channels = input_X.shape
        
        # Calculate padding 
        if self.padding == 'same':
            pad = (self.filter_size - 1)/2
        else:
            pad = 0
        
        out_h, out_w = _get_out_shape(self.filter_size,pad,self.stride,in_h,in_w)
        
        X_padded = np.pad(input_X,((0,0),(pad,pad),(pad,pad),(0,0)),'constant')
        # Zero initialization of output
        output_Z = np.zeros((batch_size,out_h,out_w,self.num_filters))
        
        for i in range(batch_size):
            a_prev_pad = X_padded[i,:,:,:]
            for h in range(out_h):
                for w in range(out_w):
                    for c in range(self.num_filters):
                        #Find the corners of the current "slice"
                        vert_start,vert_end,horiz_start,horiz_end = \
                        _get_corners(h,w,self.filter_size,self.stride)
                        
                        #Define slice
                        a_slice_prev = \
                        a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                        
                        #Conv single step
                        output_Z[i,h,w,c] = \
                        self.conv_one_step(a_slice_prev,
                                           self.W[c,:,:,:],
                                           self.b[c,:,:,:])
                        
        assert(output_Z.shape == (batch_size,out_h,out_w,self.num_filters))
        
        #Save information for backpropagation
        hparameters = {'pad': pad,
                       'stride': self.stride}
        cache = (input_X, self.W, self.b, hparameters)
        
        return output_Z, cache
                        
        
    def conv_one_step(self,a_slice_prev, W, b):
        """
        Arguments:
            a_slice_prev: (f,f, num_channels)
            W: (f,f, num_channels)
            b: (1,1,1)
        Returns:
            Z: a scalar value
        """
        s = a_slice_prev*W
        Z = np.sum(s)
        Z = Z + float(b)
        
        return Z
                        
class Pool:
    """Simple pooling layer
    
    Arguments:
        pool_size: int, specifying the height and width of the pooling window
        stride: int
        mode: str, one of ['max', 'average']
    
    """
    
    def __init__(self,pool_size, stride, mode = 'max'):
        self.pool_size = pool_size
        self.stride = stride
        self.mode = mode
        
    def forward(self,input_X):
        batch_size, in_H, in_W, in_channels = input_X.shape
        
        out_H, out_W = _get_out_shape(self.pool_size,0,self.stride,in_H,in_W)
        out_C = in_channels
        
        #Initialize output
        out_X = np.zeros((batch_size,out_H,out_W,out_C))
        
        for i in range(batch_size):
            for h in range(out_H):
                for w in range(out_W):
                    for c in range(out_C):
                        vert_start,vert_end,horiz_start,horiz_end=\
                        _get_corners(h,w,self.pool_size,self.stride)
                        prev_slice = \
                        input_X[i,vert_start:vert_end,horiz_start:horiz_end,c]
                        
                        if self.mode =='max':
                            out_X[i,h,w,c] = np.max(prev_slice)
                        elif self.mode == 'average':
                            out_X[i,h,w,c] = np.mean(prev_slice)
        #Store cache                    
        self.cache = (input_X)
        
        assert(out_X.shape==(batch_size,out_H,out_W,out_C))
        
        return out_X
  
    
def _get_out_shape(filter_size,padding,stride,in_h,in_w):
    
    """Calculate output shape
    """
    out_h = int((in_h + 2*padding-filter_size)/stride + 1)
    out_w = int((in_w + 2*padding-filter_size)/stride + 1)
    
    return out_h,out_w

def _get_corners(height,width,filter_size,stride):
    """Find the corners of the current "slice"
    """
    #Find the corners of the current "slice"
    vert_start = height*stride
    vert_end = vert_start + filter_size
    horiz_start = width*stride
    horiz_end = horiz_start + filter_size
    return vert_start,vert_end,horiz_start,horiz_end
