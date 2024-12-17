# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 19:20:36 2024

@author: 7muha
"""
from debugInitializeWeights import debugInitializeWeights
from nnComputeCost import nnCostFunction
from computeNumericalgradient import computeNumericalgradient 
import numpy as np


def checkNNGradients(lambdas=0):
   #CHECKNNGRADIENTS Creates a small neural network to check the
   #backpropagation gradients
   #   CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
   #   backpropagation gradients, it will output the analytical gradients
   #   produced by your backprop code and the numerical gradients (computed
   #   using computeNumericalGradient). These two gradient computations should
   #   result in very similar values.
   #
        
    input_layer_size = 3;
    hidden_layer_size = 5;
    hidden_layer_2_size = 5;
    num_labels = 3;
    m = 5;
   # We generate some 'random' test data
    
    Theta1 = (debugInitializeWeights(hidden_layer_size, input_layer_size))/10;
    Theta2 = (debugInitializeWeights(hidden_layer_2_size, hidden_layer_size))/10;
    Theta3 = (debugInitializeWeights(num_labels, hidden_layer_2_size))/10;
    # Unroll parameters 
    nn_params = np.expand_dims(np.concatenate((Theta1.ravel(order='F'),
                                               Theta2.ravel(order='F'),Theta3.ravel(order='F'))),axis=1);
    
   # Reusing debugInitializeWeights to generate X
    X  = debugInitializeWeights(m, input_layer_size - 1);
    y  = 1 + np.arange(1, m+1)% num_labels;
    
   # Unroll parameters
    #nn_params = np.hstack((Theta1.ravel(),Theta2.ravel()));
    
   # Short hand for cost function
    costFunc = lambda p: nnCostFunction( p, input_layer_size, hidden_layer_size,hidden_layer_2_size, num_labels,X, y, lambdas);
    
    (cost, grad) = costFunc(nn_params);
    numgrad = computeNumericalgradient(costFunc, nn_params);
    
   # Visually examine the two gradient computations.  The two columns
   # you get should be very similar. 
    print(np.column_stack((numgrad, grad)));
    print(f'The above two columns you get should be very similar.\n'+ 
             '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n');
    
   # Evaluate the norm of the difference between two solutions.  
   # If you have a correct implementation, and assuming you used EPSILON = 0.0001 
   # in computeNumericalGradient.m, then diff below should be less than 1e-9
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad);
    
    print('If your backpropagation implementation is correct, then \n' +
             'the relative difference will be small (less than 1e-9). \n' +
             f'\nRelative Difference:{diff}\n');