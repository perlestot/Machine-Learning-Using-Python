# -*- coding: utf-8 -*-

# dev_func.py

#!/usr/bin/env python3 

# Basic library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

# Developing Model 
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Fine-tuning Model
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


def build_model(total_comment = df_all['preprocessed'].values, n_hidden_layer=1, n_neurons=20, optimizer_mode='Adam', dropout_rate=0.2, recurrent_dropout_rate=0.2): 
    
    # Find the number of unique words in the data
    total_word = []
    for sent in total_comment:
        total_word.extend(sent.split())
    max_feature = len(set(total_word))
    max_length = max([len(sent.split()) for sent in total_comment]) # Equal to X.shape[0]

    tokenizer = Tokenizer(num_words=max_feature, split=' ')
    tokenizer.fit_on_texts(total_comment)
    X = tokenizer.texts_to_sequences(total_comment)
    X = pad_sequences(X)
    X.shape
    
    model = Sequential()
    model.add(Embedding(input_dim = max_feature, 
                        output_dim = 128, 
                        input_length = X.shape[1]))
    model.add(LSTM(units = 128, 
                   dropout = dropout_rate, 
                   recurrent_dropout = recurrent_dropout_rate
                  ))
    
    options = {"input_dim": X.shape[0]}
    for layer in range(n_hidden_layer):
        model.add(Dense(n_neurons, activation="relu", **options))
        options = {} 
        
    model.add(Dense(2,activation='softmax'))
    
    model.compile(loss = 'categorical_crossentropy', 
                      optimizer = optimizer_mode, 
                      metrics=['accuracy']
                     )
    return model

from sklearn.model_selection import train_test_split

def DL_HPTune(X, y, model , param_dist, 
              class_weights, n_combination, n_cv, seed,
              n_epochs, val_split):
    
    X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=val_split, random_state=seed, stratify=y)
    early_stopping_cb = EarlyStopping(monitor='val_loss', patience=10, min_delta = 0.001, restore_best_weights=True)
    
    rs_cv = RandomizedSearchCV(estimator = model, 
                               param_distributions = param_dist,  
                               scoring = "balanced_accuracy",
                               random_state = seed,
                               n_iter = n_combination, 
                               cv = n_cv,
                               n_jobs = -1,
                               iid = True
                              )
    
    print("-"*20, "RandomizedSearchCV", "-"*20)
    start = time()
    rs_cv.fit(X_trn, y_trn, epochs = n_epochs,
              class_weight = class_weights, 
              callbacks=[early_stopping_cb],
              validation_data = (X_val, y_val),
              #validation_split = val_split,
              verbose = 1)

    print("RandomizedSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(rs_cv.cv_results_['params'])))
    
    # Best parameter from RandomizedSearchCV
    
    bs_nlay = rs_cv.best_params_['n_hidden_layer']
    bs_nneu = rs_cv.best_params_['n_neurons']
    bs_optm = rs_cv.best_params_['optimizer_mode']
    bs_batch = rs_cv.best_params_['batch_size']
    
    param_grid = dict(n_hidden_layer = np.array([bs_nlay]),
                  n_neurons = np.array([bs_nneu]) ,
                  optimizer_mode = np.array([bs_optm]),
                  batch_size = np.array([bs_batch, 8*(2**(np.log2(bs_batch/8)+1))], dtype=np.int32)
                 )
    
    gs_cv = GridSearchCV(estimator = model, 
                         param_grid = param_grid,  
                         scoring = "balanced_accuracy",
                         cv = n_cv,
                         n_jobs = -1,
                         iid = True
                        )
    start = time()
    print("-"*20, "GridSearchCV", "-"*20)
    gs_cv.fit(X_trn, y_trn, epochs = n_epochs,
              class_weight = class_weights, 
              callbacks=[early_stopping_cb],
              validation_data = (X_val, y_val),
              #validation_split = val_split,
              verbose = 1)
    
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
              % (time() - start, len(gs_cv.cv_results_['params'])))
    
    return rs_cv, gs_cv