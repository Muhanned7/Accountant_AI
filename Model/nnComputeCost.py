 # -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 00:50:26 2024

@author: 7muha
"""
import numpy as np
from sigmoid import sigmoid
from SigmoidGradient import sigmoidGradient
import math
def nnCostFunction(nn_params, input_layer_size, hidden_layer_size,hidden_layer_2_size, num_labels, X, y, lambdas):
#NNCOSTFUNCTION Implements the neural network cost function for a two layer
#neural network which performs classification
#   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
#   X, y, lambda) computes the cost and gradient of the neural network. The
#   parameters for the neural network are "unrolled" into the vector
#   nn_params and need to be converted back into the weight matrices.
#
#   The returned parameter grad should be a "unrolled" vector of the
#   partial derivatives of the neural network.
#

    
   
# Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
# for our 2 layer neural network
    
    Theta1 = nn_params[:(hidden_layer_size) * (input_layer_size + 1)].copy().reshape(hidden_layer_size, (input_layer_size + 1),order='F')

    start_index =  (input_layer_size+1) * (hidden_layer_size)
    end_index =  start_index + (hidden_layer_2_size * (hidden_layer_size + 1))
    Theta2 = nn_params[start_index :end_index].copy().reshape(hidden_layer_2_size, hidden_layer_size + 1,order='F')
    Theta3 = nn_params[end_index :].copy().reshape(num_labels, hidden_layer_2_size + 1,order='F')
    # Setup some useful variables
    m = X.shape[0];
  
    # You need to return the following variables correctly
    J = 0;
    Theta1_grad = np.zeros(Theta1.shape);
    Theta2_grad = np.zeros(Theta2.shape);
    Theta3_grad= np.zeros(Theta3.shape);
    delta_2 = 0;
    delta_3 =0;
    delta_4= 0;
    delta_1= 0;
# ====================== YOUR CODE HERE ======================
# Instructions: You should complete the code by working through the
#               following parts.
#
# Part 1: Feedforward the neural network and return the cost in the
#         variable J. After implementing Part 1, you can verify that your
#         cost function computation is correct by verifying the cost
#         computed in ex4.m
    a_1 = np.ones(X[1,:].size);
    a_1 = np.expand_dims(a_1,axis=1);    
    a_2 = np.ones(hidden_layer_size);
    a_2 = np.expand_dims(a_2,axis=1);
    a_3 = np.ones(hidden_layer_2_size);
    a_3 = np.expand_dims(a_3,axis=1);
    a_4 = np.ones(num_labels);  
    y_vect = np.zeros(num_labels);
    
#    y_vect(y(i,:)) = 1
    for i in range(m):
        y_vect = np.zeros(num_labels)
        y_vect[int(y[i]) - 1] = 1;
        y_vect = np.expand_dims(y_vect,axis=1);
        a_1 = np.expand_dims(X[i,:],axis=1);
        g_2 = np.dot(Theta1,np.vstack((1,a_1)));
        a_2 = sigmoid(g_2);
        g_3 = np.dot(Theta2,np.vstack((1,a_2)));
        a_3 = sigmoid(g_3);
        g_4 = np.dot(Theta3,np.vstack((1,a_3)));
        a_4=  sigmoid(g_4);
        tilda_4 = a_4 - y_vect;
        tilda_3 = (np.dot(Theta3.T,tilda_4))*(np.vstack((1,(a_3*(1-a_3)))));
        tilda_2 = (np.dot(Theta2.T,tilda_3[1:]))*(np.vstack((1,(a_2*(1-a_2)))));
        delta_1 = delta_1 + np.dot(tilda_2, (np.vstack((1,a_1))).T);
        delta_2 = delta_2 + np.dot(tilda_3, (np.vstack((1,a_2))).T);
        delta_3 = delta_3 + np.dot(tilda_4, (np.vstack((1,a_3))).T)
        J = J + sum(-(y_vect * np.log(a_4)) - ((1-y_vect) * np.log(1-a_4)));
          
    #regularization factor

    J = ((1/m) * J) + ((lambdas/(2*m)) * (np.sum(Theta1[:,1:]**2) + np.sum(Theta2[:,1:]**2) + np.sum(Theta3[:,1:]**2)));
    Theta3[:,0:1] = np.zeros((Theta3.shape[0],1));
    Theta3[:,1:] = ((lambdas/m) * Theta3[:,1:]);
    Theta3_grad = ((1/m) * delta_3) +  Theta3;
    Theta2[:,0:1] = np.zeros((Theta2.shape[0],1));
    Theta2[:,1:] = ((lambdas/m) * Theta2[:,1:]);
    Theta2_grad = ((1/m) * delta_2[1:]) +  Theta2;
    Theta1[:,0:1] = np.zeros((Theta1.shape[0],1));
    Theta1[:,1:] = ((lambdas/m) * Theta1[:,1:]);
    Theta1_grad = ((1/m) * delta_1[1:]) + Theta1;


# -------------------------------------------------------------

# =========================================================================

# Unroll gradients
    grad = np.vstack((Theta1_grad.reshape(-1,1,order='F'), Theta2_grad.reshape(-1, 1,order='F'),Theta3_grad.reshape(-1,1,order='F')));
    grad = grad.flatten();
    return J, grad 