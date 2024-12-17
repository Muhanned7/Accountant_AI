# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 00:59:42 2024

@author: 7muha
"""

import numpy as np
def sigmoid(z): 
    g = 1.0 / (1.0 + np.exp(-z));
    return g