# -*- coding: utf-8 -*-

# prepare_data.py

#!/usr/bin/env python3 


# Basic library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data preprocessing
import re
import contractions
import unicodedata
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Prepare the data
def pre_process(text):
    
    # Normalize text to same unicode
    # Replacing non-breaking space ('\xc2\xa0') with a space.
    text = unicodedata.normalize("NFKD", text)
    
    # Lower Case
    text = text.lower()
    
    # Removing Punctuation
    text = re.sub(r'[^\w\s]','',text)
    
    # Expand contraction
    text = contractions.fix(text)
    
    # Removal of Stop Words
    text = " ".join(x for x in text.split() if x not in stop_words)
    
    # Word Lemmatization
    text = " ".join(lemmatizer.lemmatize(x) for x in text.split())
    
    return text

# Split data to train and test
def split_data(X, y, seed):
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    print("Size of training data is " + str(X_trn.shape))
    #y_trn_bin = to_categorical(y_trn)
    print("Size of training label is " + str(y_trn.shape))
    unique, counts = np.unique(y_trn, return_counts=True)
    print('Class Counts:',np.asarray((unique, counts)).T.ravel())
    print()
    print("Size of testing data is " + str(X_tst.shape))
    #y_tst_bin = to_categorical(y_tst)
    print("Size of testing label is " + str(y_tst.shape))
    unique, counts = np.unique(y_tst, return_counts=True)
    print('Class Counts:',np.asarray((unique, counts)).T.ravel())
    print()
    unique, counts = np.unique(y, return_counts=True)
    print('Total Class Counts:',np.asarray((unique, counts)).T.ravel())
    return X_trn, X_tst, y_trn, y_tst