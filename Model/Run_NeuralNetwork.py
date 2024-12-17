# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 19:24:42 2024

@author: 7muha
"""
import numpy as np
from scipy.optimize import minimize, fmin_cg, fmin_bfgs
from SigmoidGradient import sigmoidGradient
from nnComputeCost import nnCostFunction
from checkNNGradients import checkNNGradients
from predict import predict
from computeNumericalgradient import computeNumericalgradient
from xavier_init import xavier_init
from Adam import adam_optimize

def Run_NeuralNetwork(X,y, X_validation, Y_validation):
    # Machine Learning Online Class - Exercise 4 Neural Network Learning
    
    #  Instructions
    #  ------------
    #  This file contains code that helps you get started on the
    #  linear exercise. You will need to complete the following functions 
    #  in this exericse:
    #
    #     sigmoidGradient.m
    #     randInitializeWeights.m
    #    nnCostFunction.m
    #
    #  For this exercise, you will not need to change any code in this file,
    #  or any other files other than those mentioned above.
    #
    
    # Initialization
    #clear ; close all; clc
    
    # Setup the parameters you will use for this exercise
    input_layer_size  = 3;  # 20x20 Input Images of Digits
    hidden_layer_size = 5;   # 25 hidden units
    hidden_layer_2_size = 7;
    num_labels = 9;          # 10 labels, from 1 to 10   
                              # (note that we have mapped "0" to label 10)
    
    ## =========== Part 1: Loading and Visualizing Data =============
    #  We start the exercise by first loading and visualizing the dataset. 
    #  You will be working with a dataset that contains handwritten digits.
    #
    
    # Load Training Data
    
    m = X.shape[0];
    
    # Randomly select 100 data points to display
    sel = np.random.permutation((np.size(X,0)))
    
    
    ## ================ Part 2: Loading Parameters ================
    # In this part of the exercise, we load some pre-initialized 
    # neural network parameters.
    
    Theta1 = xavier_init([hidden_layer_size, input_layer_size+1],hidden_layer_size, input_layer_size+1);
    Theta2 = xavier_init([hidden_layer_2_size, hidden_layer_size+1],hidden_layer_2_size, hidden_layer_size+1);
    Theta3 = xavier_init([num_labels, hidden_layer_2_size+1],num_labels, hidden_layer_2_size+1);
    # Unroll parameters 
    nn_params = np.expand_dims(np.concatenate((Theta1.ravel(order='F'),
                                               Theta2.ravel(order='F'),Theta3.ravel(order='F'))),axis=1);
    ## ================ Part 3: Compute Cost (Feedforward) ================
    #  To the neural network, you should first start by implementing the
    #  feedforward part of the neural network that returns the cost only. You
    #  should complete the code in nnCostFunction.m to return cost. After
    #  implementing the feedforward to compute the cost, you can verify that
    #  your implementation is correct by verifying that you get the same cost
    #  as us for the fixed debugging parameters.
    #
    #  We suggest implementing the feedforward cost *without* regularization
    #  first so that it will be easier for you to debug. Later, in part 4, you
    #  will get to implement the regularized cost.
    #
    print('\nFeedforward Using Neural Network ...\n')
    
    # Weight regularization parameter (we set this to 0 here).
    lambdas = 0;
    
    (J,grad) = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,hidden_layer_2_size,
                       num_labels, X, y, lambdas);
    
    print(f'Cost at parameters (loaded from ex4weights): {J} '
             '\n(this value should be about 0.287629)\n');
    
    print('\nProgram paused. Press enter to continue.\n');
    
    ## =============== Part 4: Implement Regularization ===============
    #  Once your cost function implementation is correct, you should now
    #  continue to implement the regularization with the cost.
    #
    
    print('\nChecking Cost Function (w/ Regularization) ... \n')
    
    # Weight regularization parameter (we set this to 1 here).
    lambdas = 1;
    
    (J,grad) = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,hidden_layer_2_size,
                       num_labels, X, y, lambdas);
   
    print(f"Cost at parameters (loaded from ex4weights): {J} "
           '\n(this value should be about 0.383770)\n');
    
    input('Program paused. Press enter to continue.\n');
    
    
    
    ## ================ Part 5: Sigmoid Gradient  ================
    #  Before you start implementing the neural network, you will first
    #  implement the gradient for the sigmoid function. You should complete the
    #  code in the sigmoidGradient.m file.
    #
    
    print('\nEvaluating sigmoid gradient...\n');
    
    g = sigmoidGradient(np.array([-1, -0.5, 0, 0.5, 1]));
    print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ');
    print( g);
    print('\n\n');
    
    print('Program paused. Press enter to continue.\n');
    
    
    
    ## ================ Part 6: Initializing Pameters ================
    #  In this part of the exercise, you will be starting to implment a two
    #  layer neural network that classifies digits. You will start by
    #  implementing a function to initialize the weights of the neural network
    #  (randInitializeWeights.m)
    
    print('\nInitializing Neural Network Parameters ...\n')
    np.random.seed(15)
    Theta1 = (np.random.rand(hidden_layer_size,X.shape[1] +1))/10;
    Theta2 = (np.random.rand(hidden_layer_2_size,hidden_layer_size+1))/10;
    Theta3 = (np.random.rand(num_labels,hidden_layer_2_size+1))/10;
    # Unroll parameters 
    initial_nn_params = np.vstack((Theta1.reshape(Theta1.size,1),
                           Theta2.reshape(Theta2.size,1),Theta3.reshape(Theta3.size,1)));
    
    
    ## =============== Part 7: Implement Backpropagation ===============
    #  Once your cost matches up with ours, you should proceed to implement the
    #  backpropagation algorithm for the neural network. You should add to the
    #  code you've written in nnCostFunction.m to return the partial
    #  derivatives of the parameters.
    #
    print('\nChecking Backpropagation... \n');
    
    #  Check gradients by running checkNNGradients
    checkNNGradients();
    
    print('\nProgram paused. Press enter to continue.\n');
    
    
    
    ## =============== Part 8: Implement Regularization ===============
    #  Once your backpropagation implementation is correct, you should now
    #  continue to implement the regularization with the cost and gradient.
    #
    
    print('\nChecking Backpropagation (w/ Regularization) ... \n');
    
    #  Check gradients by running checkNNGradients
    lambdas = 3;
    #checkNNGradients(lambdas);
    
    # Also output the costFunction debugging values
    (debug_J,grad)  = nnCostFunction(nn_params, input_layer_size,
                              hidden_layer_size, hidden_layer_2_size , num_labels, X, y, lambdas);
    
    print(f'\n\nCost at (fixed) debugging parameters (w/ lambda = {lambdas}): {debug_J} ' +  
             '\n(for lambda = 3, this value should be about 0.576051)\n\n');
    
    print('Program paused. Press enter to continue.\n');
    
    
    ## =================== Part 8: Training NN ===================
    #  You have now implemented all the code necessary to train a neural 
    #  network. To train your neural network, we will now use "fmincg", which
    #  is a function which works similarly to "fminunc". Recall that these
    #  advanced optimizers are able to train our cost functions efficiently as
    #  long as we provide them with the gradient computations.
    #
    print('\nTraining Neural Network... \n')
    
    #  After you have completed the assignment, change the MaxIter to a larger
    #  value to see how more training helps.
    options = {
    'disp': True,  # Display convergence messages
    'maxiter': 1000,  # Maximum number of iterations
    'gtol': 1e-6  # Gradient tolerance for termination
    }
    
    #  You should also try different values of lambda
    lambdas = 10;
    
    # Create "short hand" for the cost function to be minimized
    costFunction =  lambda p: nnCostFunction( p, input_layer_size, hidden_layer_size,hidden_layer_2_size, num_labels,
            X, y, lambdas);
    
    initial_nn_params = initial_nn_params.flatten();
    # Now, costFunction is a function that takes in only one argument (the
    # neural network parameters)
    #(cost,nn_params) = fmincg(costFunction, initial_nn_params, options);
    #result_fmin_cg = fmin_cg(lambda p: costFunction(p)[0], initial_nn_params, fprime=lambda p: costFunction(p)[1], maxiter=1000, disp=True);     
    #result_fmin_cg = fmin_bfgs(lambda p: costFunction(p)[0], initial_nn_params, fprime=lambda p: costFunction(p)[1], maxiter=2000, disp=True)
    #result_fmin_cg = fmin_bfgs(lambda p: costFunction(p)[0], initial_nn_params, fprime=lambda p: costFunction(p)[1], maxiter=2000, disp=True)
    result_Adam   = adam_optimize(lambda p:costFunction(p)[0], lambda p:costFunction(p)[1], initial_nn_params, lr=0.001,
                                  beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=1000)
    # Obtain Theta1 and Theta2 back from nn_params
    print(result_Adam.shape,nn_params.shape)
    
    # Run it once with validation set data.
    
    costFunction =  lambda p: nnCostFunction( p, input_layer_size, hidden_layer_size,hidden_layer_2_size, num_labels,
            X_validation, Y_validation, lambdas);
    result_Adam   = adam_optimize(lambda p:costFunction(p)[0], lambda p:costFunction(p)[1], result_Adam, lr=0.001,
                                  beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=1)
    Theta1 = result_Adam[:(hidden_layer_size) * (input_layer_size + 1)].copy().reshape(hidden_layer_size, (input_layer_size + 1),order='F')

    start_index =  (input_layer_size+1) * (hidden_layer_size)
    end_index =  start_index+(hidden_layer_size+1) * (hidden_layer_2_size)
    Theta2 = result_Adam[start_index :end_index].copy().reshape(hidden_layer_2_size, hidden_layer_size + 1,order='F')
    Theta3 = result_Adam[end_index :].copy().reshape(num_labels, hidden_layer_2_size + 1,order='F')
  
    input('Program paused. Press enter to continue.\n');
    
    

    ## ================= Part 9: Visualize Weights =================
    #  You can now "visualize" what the neural network is learning by 
    #  displaying the hidden units to see what features they are capturing in 
    #  the data.
    
    #fprintf('\nVisualizing Neural Network... \n')
    
    #displayData(Theta1(:, 2:end));
    
    #print('\nProgram paused. Press enter to continue.\n');
    #pause;
    
    ## ================= Part 10: Implement Predict =================
    #  After training the neural network, we would like to use it to predict
    #  the labels. You will now implement the "predict" function to use the
    #  neural network to predict the labels of the training set. This lets
    #  you compute the training set accuracy.
    
    pred = predict(Theta1, Theta2, Theta3,X);
    print("prediction: ", pred);
    return (Theta1, Theta2, Theta3)
    #print('\nTraining Set Accuracy: #f\n', mean(double(pred == y)) * 100);
    
    
