# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 21:36:22 2024

@author: 7muha
"""
import csv
import Compute_Cost
import Gradient_Descent
import matplotlib.pyplot as plt
import Plot_Graph
import numpy as np
import pandas as pd
from nnComputeCost import nnCostFunction
from predict import predict
import json
from Run_NeuralNetwork import Run_NeuralNetwork

Theta1 =0
Theta2=0
Theta3=0




# open the files and read content
with open('extracted_data_training_new.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    # Some gradient descent settings
    iterations = 1500;
    alpha = 0.01;
    # Optionally skip the header row
    next(csv_reader, None)
    X = [];
    Y = [];
    # Read and print each row
    for row in csv_reader:
        X.append(row[0:3]);
        Y.append(row[3]);
# seperate the data and the output
    Theta =[0, 0, 0, 0];
    (X,Y,Theta) = Compute_Cost.Clean_Data(X, Y, Theta); 
    X[:,0] = (Compute_Cost.Normalize_features(X[:,0]))/10
    X[:,1] = Compute_Cost.Normalize_features(X[:,1])
    X[:,2] = Compute_Cost.Normalize_features(X[:,2])
    iterations =1500
    alpha = 0.01
    J_test = Compute_Cost.Compute_Cost(X, Y, Theta)
    Theta = Gradient_Descent.Gradient_Descent(X, Y, Theta, alpha, iterations);
    X_training = X[:int(0.60 * len(X))]
    Y_training = Y[:int(0.60 * len(Y))]
    X_validation = X[int(0.60 * len(X)):]
    Y_validation = Y[int(0.60 * len(X)):]
    print('X: ',  len(X), 'X_Training: ',len(X_training),'Y_Training: ',len(Y_training),
          'X_Validation: ',len(X_validation), 'Y_validation', len(Y_validation))
    
    # Create subplots for each feature of X against Y
    for i in range(0,3):
        # Select two features from X to plot (e.g., feature 1 and feature 2) 
        if i ==2:
            z =0
        else:
            z = i+1
        
        if i == 0:
            label_1 = "Amount"
            label_2 = "Date"
        elif i==1:
            label_1 = "Description"
            label_2 = "Amount"
        else:
            label_1= "Date"
            label_2 = "Description"
        x_feature_1 = X[:, z]  # First feature of X
        x_feature_2 = X[:, i]  # Second feature of X
        #Plot_Graph.PlotGraph(x_feature_1, label_1, x_feature_2,label_2, Y)
        Plot_Graph.Plot_2_Graph(x_feature_1, label_1, x_feature_2, label_2,Y)
        np.savetxt('X_matrix.csv', X, delimiter=',', fmt='%.5f')
        np.savetxt('Y_matrix.csv', Y, delimiter=',', fmt='%.5f')
    (Theta1,Theta2,Theta3) = Run_NeuralNetwork(X_training,Y_training, X_validation,Y_validation);
    print("Theta1 = ",Theta1.shape,Theta1.tolist())
    Theta={ "Theta1": Theta1.tolist(), "Theta2":Theta2.tolist(), "Theta3":Theta3.tolist()}
    output_file = "theta_values.json"
    with open(output_file, mode='w') as file:
        json.dump(Theta, file, indent=4)
        print(f"Theta values saved to {output_file}")
    #J = nnCostFunction(input_layer_size, hidden_layer_size, num_labels, X, Y, lambdas)

    
    
    
    
   