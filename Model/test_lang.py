# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 20:05:34 2024

@author: 7muha
"""

import nltk
import scipy
from nltk.tokenize import word_tokenize
import numpy as np
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data (a list of sentences)
corpus = [
    "I love programming in Python",
    "Python is a great language",
    "I enjoy learning new algorithms",
    "Machine learning is fascinating",
    "Deep learning and neural networks are powerful"
]

# Initialize the Tokenizer
tokenizer = Tokenizer()

# Fit the tokenizer on the sentences
tokenizer.fit_on_texts(corpus)

# Convert sentences to sequences of integers
sequences = tokenizer.texts_to_sequences(corpus)

# View the word index (optional)
word_index = tokenizer.word_index
print("Word Index:", word_index)

# View the sequences
print("Sequences:", sequences)


