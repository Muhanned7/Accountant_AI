# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 04:00:25 2024

@author: 7muha
"""
import numpy as np


def adam_optimize(f, grad_f, theta, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=1000):
    """
    Adam optimizer implementation for a given objective function and its gradient.
    
    Parameters:
    f         : Objective function to minimize
    grad_f    : Function to compute the gradient of the objective function
    theta     : Initial parameter values (NumPy array)
    lr        : Learning rate (default: 0.001)
    beta1     : Exponential decay rate for the first moment estimates (default: 0.9)
    beta2     : Exponential decay rate for the second moment estimates (default: 0.999)
    epsilon   : A small constant to avoid division by zero (default: 1e-8)
    max_iter  : Maximum number of iterations (default: 1000)
    
    Returns:
    theta     : The optimized parameters after `max_iter` iterations
    """
    
    m = np.zeros_like(theta)  # First moment vector (mean of the gradient)
    v = np.zeros_like(theta)  # Second moment vector (uncentered variance of the gradient)
    
    for t in range(1, max_iter + 1):
        # Compute the gradient of the objective function
        grad = grad_f(theta)
        
        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * grad
        
        # Update biased second moment estimate
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - beta1 ** t)
        
        # Compute bias-corrected second moment estimate
        v_hat = v / (1 - beta2 ** t)
        
        # Update parameters
        theta -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
        
        # Optional: Print progress every 100 iterations
        if t % 100 == 0:
            print(f"Iteration {t}: f(theta) = {f(theta)}")
    
    return theta