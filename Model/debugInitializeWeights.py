# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:55:53 2024

@author: 7muha
"""
import numpy as np
import math

def debugInitializeWeights(fan_out, fan_in):
    #DEBUGINITIALIZEWEIGHTS Initialize the weights of a layer with fan_in
    #incoming connections and fan_out outgoing connections using a fixed
    #strategy, this will help you later in debugging
    #   W = DEBUGINITIALIZEWEIGHTS(fan_in, fan_out) initializes the weights 
    #   of a layer with fan_in incoming connections and fan_out outgoing 
    #   connections using a fix set of values
    #
    #   Note that W should be set to a matrix of size(1 + fan_in, fan_out) as
    #   the first row of W handles the "bias" terms
    #
    
    # Set W to zeros
    W = np.zeros((fan_out, 1 + fan_in));
    
    # Initialize W using "sin", this ensures that W is always of the same
    # values and will be useful for debugging
    num_elements = W.size

    # Create an array from 1 to num_elements
    indices = np.arange(1, num_elements + 1)

    # Apply the sine function to the array of indices
    sin_values = np.sin(indices) /10

    W = (sin_values.reshape(W.shape, order='F'))
    
    return W
    
    # =========================================================================
