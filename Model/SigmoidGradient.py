# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 18:34:23 2024

@author: 7muha
"""
from sigmoid import sigmoid

def sigmoidGradient(z):
    g = sigmoid(z) * (1-sigmoid(z))
    return g