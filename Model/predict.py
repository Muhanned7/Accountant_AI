# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:43:07 2024

@author: 7muha
"""
import numpy as np
from sigmoid import sigmoid
def predict(Theta1, Theta2,Theta3, X):
    #PREDICT Predict the label of an input given a trained neural network
    #   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    #   trained weights of a neural network (Theta1, Theta2)
    
    # Useful values
    m =  X.shape[0];
    num_labels = Theta2.shape[0];
    
    # You need to return the following variables correctly 
    p = np.zeros((X.shape[0], 1));
    
    h1 = sigmoid(np.dot(np.hstack((np.ones((m, 1)), X)) , Theta1.T));
    h2 = sigmoid(np.dot(np.hstack((np.ones((m, 1)), h1)) , Theta2.T));
    h3 = sigmoid(np.dot(np.hstack((np.ones((m, 1)), h2)) , Theta3.T));
    p = np.argmax(h3, axis=1);
    return p
    
    # =========================================================================
    
    
    