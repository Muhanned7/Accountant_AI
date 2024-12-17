# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:59:39 2024

@author: 7muha
"""

import numpy as np
from scipy.optimize import fmin_cg

# Define cost function that returns both cost and gradient
def cost_function(theta):
    theta = theta.flatten()  # Ensure theta is 1D
    cost = np.sum(theta**2)  # Example: sum of squares as cost
    gradient = 2 * theta     # Gradient is 2*theta
    return cost, gradient


cost_Func = lambda p: cost_function(p)
params =np.random.rand(61)
# Define the lambda for optimization
result = fmin_cg(lambda theta: cost_Func(theta)[0], params,                 
                 fprime=lambda theta: cost_function(theta)[1],  # Gradient as 1D
                 maxiter=200, disp=True)

print("Optimal theta:", result)