# -*- coding: utf-8 -*-

# eval_report.py

#!/usr/bin/env python3 


# Basic library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from itertools import cycle
Fignum = cycle(range(1,20))

# Evaluation
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.externals import joblib

# Plot ROC Curve
def plot_roc(X_test, y_test, GridSearchCV_List, tittle=''):
    
    # Define cycol for iteration color
    from itertools import cycle
    cycol = cycle('bgrcmyk')
    cycol2 = cycle('bgrcmyk')
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title('Figure '+ str((next(Fignum))) + ': ROC curve ' + tittle)  
    
    for i in GridSearchCV_List:
         
        P_true = eval(i+'.predict_proba(X_test)[:, 1]')
        fpr, tpr, thresholds = roc_curve(y_test, P_true, pos_label=1)
        auc = round(roc_auc_score(y_test, P_true), 2)

        ax.plot(fpr, tpr, color=next(cycol), label=i[:-3]+", auc=" + str(auc))
        ax.fill_between(fpr, 0, tpr, color=next(cycol2), alpha=0.2)
        ax.plot(ax.get_ylim(), ax.get_xlim(), color="gray", linestyle=':',linewidth=1.5)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_xlim(left=0, right=1)
        ax.set_ylim(bottom=0, top=1)
        plt.legend(loc="best")
    
    plt.show()    

# Plot PR Curve 
def plot_pr(X_test, y_test, GridSearchCV_List, lab =1, tittle=''):
    
    # Define color for iteration color
    from itertools import cycle
    cycol = cycle('bgrcmyk')
    cycol2 = cycle('bgrcmyk')
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title('Figure '+ str((next(Fignum))) + ': PR curve ' + tittle)
    
    for i in GridSearchCV_List:
        
        P_true = eval(i+'.predict_proba(X_test)[:, 1]')
        ap = round(average_precision_score(y_test, P_true, pos_label=lab), 2)
        precision, recall, thresholds = precision_recall_curve(y_test, P_true, pos_label=lab)
        
        ax.plot(recall, precision, color=next(cycol), label=i[:-3]+", average precision=" + str(ap))
        ax.fill_between(recall, 0, precision, color=next(cycol2), alpha=0.2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(left=0, right=1)
        ax.set_ylim(bottom=0, top=1)
        plt.legend(loc="best")

    plt.show()    


# Make a model summary report
def model_report(X_test,y_test,GridSearchCV_List,model_func):

    model_list = []
    accuracy_list = []
    balanced_acc_list = []
    auc_list = []
    cm_list = []
    recall_list = []
    precision_list = []
    ap_list = []
    f1_list = []
    best_params_list = []
    

    for i,j in enumerate(GridSearchCV_List):
        locals()[j] = model_func[i]
        
        if isinstance(model_func[i], 
                      (sklearn.model_selection._search.RandomizedSearchCV,
                       sklearn.model_selection.GridSearchCV)):            
            y_pred = eval(j+'.best_estimator_.predict(X_test)')
            P_true = eval(j+'.predict_proba(X_test)[:, 1]')
        else:
            y_pred = eval(j+'.predict_classes(X_test)')
            P_true = eval(j+'.predict_proba(X_test)[:, 1]')
             
        model_name = j
        TP, FN, FP, TN = confusion_matrix(y_test, y_pred, labels=[1, 0]).ravel()
        
        Accuracy = round(accuracy_score(y_test, y_pred), 2)
        bal_acc = round(balanced_accuracy_score(y_test, y_pred),2)
        auc = round(roc_auc_score(y_test, P_true), 2)
        cm = dict(TP=TP, FP=FP, FN=FN, TN=TN)
        Recall = round(recall_score(y_test, y_pred, average='binary'), 2)
        Precision = round(precision_score(y_test, y_pred, average='binary'), 2)
        ap = round(average_precision_score(y_test, P_true, pos_label=1), 2)
        F1_score = round(f1_score(y_test, y_pred,average='binary'), 2)
        best_params = eval(j+'.best_params_')
        
        model_list.append(model_name)
        accuracy_list.append(Accuracy)
        balanced_acc_list.append(bal_acc)
        auc_list.append(auc)
        cm_list.append(cm)
        recall_list.append(Recall)
        precision_list.append(Precision)
        ap_list.append(ap)
        f1_list.append(F1_score)
        best_params_list.append(best_params)
        

    report = dict(Model = model_list,
                  Accuracy = accuracy_list,
                  Balanced_Accuracy = balanced_acc_list,
                  AUC = auc_list,
                  Confusion_Matrix = cm_list,
                  Recall = recall_list,
                  Precision = precision_list,
                  Average_Precision = ap_list,
                  F1_score = f1_list,
                  Best_Parameters = best_params_list )
    
    df_report = pd.DataFrame.from_dict(report)
    pd.set_option('display.max_colwidth', -1)
    
    return df_report