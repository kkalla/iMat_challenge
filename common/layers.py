# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 19:05:46 2018

@author: kkalla

Building blocks for Convolutional Nueral Networks w/o TensorFlow
"""

import numpy as np

def Conv():
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
    def __init__(self,num_input_channels, filter_size = 3, padding, stride=2, num_filters,
                 initializer):
        
        self.num_input_channels = num_input_channels
        
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
        out_h, out_w = _get_out_shape(self.filter_size,self.padding,in_h,in_w)
        
        # Calculate padding 
        if self.padding == 'same':
            pad = (self.filter_size - 1)/2
        else:
            pad = 0
        
        X_padded = np.pad(input_X,((0,0),(pad,pad),(pad,pad)(0,0)),'constant')
        # Zero initialization of output
        output_Z = np.zeros((batch_size,out_h,out_w,self.num_filters))
        
        for i in range(batch_size):
            a_prev_pad = X_padded[i,:,:,:]
            for h in range(out_h):
                for w in range(out_w):
                    for c in range(num_filters):
                        #Find the corners of the current "slice"
                        vert_start = h*self.stride
                        vert_end = vert_start + self.filter_size
                        horiz_start = w*self.stride
                        horiz_end = horiz_start + self.filter_size
                        
                        #Define slice
                        a_slice_prev = \
                        a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                        #Conv single step
                        output_Z[i,h,w,c] = self.conv_one_step(
                                a_slice_prev,self.W[c,:,:,:],self.b[c,:,:,:])
                        
        assert(output_Z.shape == (batch_size,out_h,out_w,self.num_filters))
        
        #Save information for backpropagation
        hparameters = {'pad': pad,
                       'stride': self.stride}
        cache = (input_X, self.W, self.b, hparameters)
        
        return output_Z, cache
                        
        
    def conv_one_step(a_slice_prev, W, b):
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
                        
  
def _get_out_shape(filter_size,padding,in_h,in_w):
    
    """Calculate output shape
    """
    out_h = int((in_h+2*padding-filter_size)/2 + 1)
    out_w = int((in_w+2*padding-filter_size)/2 + 1)
    
    return out_h,out_w