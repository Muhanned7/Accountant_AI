# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 19:15:55 2024

@author: 7muha
"""

from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime
import numpy as np
import re


def preprocess_date(date_str):
    # If day part is single digit, pad with zero
    if len(date_str.split('-')[0]) == 1:
        date_str = '0' + date_str
    return date_str

def Clean_Data(X, y, theta):
    m = len(y); # number of training examples

# You need to return the following variables correctly
    J = 0;
    descriptions=[]
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#   You should set J to the cost.
    vectorizer = TfidfVectorizer()
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
        try:
            # Try parsing with zero-padded format
           date_obj = datetime.strptime(row[0] + "-2024", '%d-%b-%Y')
        except ValueError:
           # If it fails, try without zero-padding
           date_obj = datetime.strptime(row[0] + "-2024", '%b-%d-%Y')
   
    
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
def Compute_Cost(X,y, Theta):
    m = len(y); # number of training examples

    J =  1/(2*m) * sum((X @ Theta - y)**2);
    return J

def Normalize_features(array_1):
    X_min = array_1.min(axis=0)
    X_max = array_1.max(axis=0)
    X_normalized = (array_1 - X_min) / (X_max - X_min)
    return X_normalized
    