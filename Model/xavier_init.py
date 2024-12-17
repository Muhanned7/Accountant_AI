# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 08:20:30 2024

@author: 7muha
"""
import numpy as np
def xavier_init(size, n_in, n_out):
    return np.random.randn(size[0], size[1]) * np.sqrt(2 / (n_in + n_out))
