# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 10:21:58 2024

@author: 7muha
"""
import Compute_Cost
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.preprocessing.text import Tokenizer

from datetime import datetime

import numpy as np

import json

from predict import predict

import re

def User_classifier(User_Data):
    X = [];
    Y = [];
    for row in User_Data:
        
        X.append([str(row['date']).replace('/','-'),row['amount'],row['description']]);
        Y.append(row['type']);
# seperate the data and the output
    Theta =[0, 0, 0, 0];
    (X,Y,Theta) = Clean_Data(X, Y); 
    X[:,0] = (Compute_Cost.Normalize_features(X[:,0]))/10
    X[:,1] = Compute_Cost.Normalize_features(X[:,1])
    X[:,2] = Compute_Cost.Normalize_features(X[:,2])
    #type_list = Theta * [User_Data['date'], User_Data['amount'],User_Data['description']]
    with open("D:\\Bank_Statement\\theta_values.json", mode="r") as file:
        Theta = json.load(file)
        Theta1 = np.array(Theta.get("Theta1"))
        Theta2 = np.array(Theta.get("Theta2"))
        Theta3 = np.array(Theta.get("Theta3"))
        Y = predict(Theta1, Theta2,Theta3, X)
    count =0
    for row in User_Data:
        row['type'] = int(Y[count])
        #print(type(row['type']), type(Y[count]))
        count+=1
    return User_Data

def Clean_Data(X,y):
    
       m = len(y); # number of training examples

    # You need to return the following variables correctly
       J = 0;
       descriptions=[]
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #   You should set J to the cost.
       #vectorizer = TfidfVectorizer()
        # Initialize the Tokenizer
       tokenizer = Tokenizer()
       for desc in X:
           desc[2] =re.sub(r'\b\w*[a-zA-Z]\w*\d\w*\b|\b\w*\d\w*[a-zA-Z]\w*\b', '', desc[2])
           desc[2] = desc[2].replace('\n', ' ').replace('-', ' ')
           desc[2] = re.sub(r'\d+', '', desc[2])
           descriptions.append(desc[2])
       tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
       tokenizer.fit_on_texts(descriptions)
       for row in X:
           descriptions.append(row[2]);
    # J =  1/(2*m) * sum((X * theta - y).^2);
           row[2] = re.sub(r'\b\w*[a-zA-Z]\w*\d\w*\b|\b\w*\d\w*[a-zA-Z]\w*\b', '', row[2])
           # Fit the tokenizer on the sentences
           row[2] = [row[2]]

            
            # Convert sentences to sequences of integers
           row[2] = tokenizer.texts_to_sequences(row[2])    
        # Add a default year (e.g., 2024)
           #try:
               # Try parsing with zero-padded format
            #  date_obj = datetime.strptime(row[0] + "-2024", '%d-%b-%Y')
           #except ValueError:
              # If it fails, try without zero-padding
           date_obj = datetime.strptime(row[0] + "-2024", '%m-%d-%Y')
       
        
           day = date_obj.day
           month = date_obj.month
           amount = float(row[1].replace(',', ''))
           temp_list = sum(row[2][0])
           row[0] = int(f"{day:02}{month:02}")
           row[1] = amount
           row[2] = temp_list
       max_length = max(len(lst) for lst in X)
       X = [lst + [0] * (max_length - len(lst)) for lst in X]     
       X = np.array(X)
       theta = np.zeros(max_length)
       theta = np.array(theta, dtype=np.int64) 
       theta= theta[:,np.newaxis]  
       y = ['0' if x == '' else x for x in y]
       y = np.array(list(map(float, y)))
       y=y[:,np.newaxis]
        
            #row = pad_sequences(row, padding='post')
            #J =  1/(2*m) * sum((X * theta - y).^2);
       return (X,y,theta)