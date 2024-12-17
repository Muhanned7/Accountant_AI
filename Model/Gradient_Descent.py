# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 16:42:41 2024

@author: 7muha
"""
import Compute_Cost
import numpy as np

def Gradient_Descent(X, y, theta, alpha, num_iters):
    m = len(y); # number of training examples
    J_history = np.zeros((num_iters, 1));
    Theta = theta
    for iters in range(num_iters):
    
        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #
        
        
        Theta = Theta + ((alpha/m) * (np.sum((X @ Theta - y) * X,axis=0, dtype=np.int64) )[:,np.newaxis])

        
    
    
        # ============================================================
    
        # Save the cost J in every iteration
        
        J_history[iters] = Compute_Cost.Compute_Cost(X, y, Theta)
        
    
    
    return Theta