# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:56:59 2024

@author: 7muha
"""

import numpy as np
import math

def computeNumericalgradient(J, theta):
    numgrad = np.zeros(theta.shape);
    perturb = np.zeros(theta.shape);
    e = 1* (math.e)**-4;
    for p in range(theta.size):
        # Set perturbation vector
        perturb[p] = e;
        (loss1,grad) = J(theta - perturb);
        (loss2,grad) = J(theta + perturb);
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2*e);
        perturb[p] = 0;
    return numgrad