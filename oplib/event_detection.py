# Copyright (C) Responsive Capital Management - All Rights Reserved.
# Unauthorized copying of any code in this directory, via any medium is strictly prohibited.
# Proprietary and confidential.
# Written by Logan Grosenick <logan@responsive.ai>, 2019.

import numpy as np
from sklearn import svm, linear_model
from util.validation import balance_binary_data
from screening import med_corr_select
import sklearn.preprocessing as preprocessing

def event_classifier(clf, X, y, CV_folds, select_idx=None, transformer=None,
                    rescale=False, sample_weights=None):
    
    if rescale:
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(X)
        
    n = X.shape[0] 
    y_shuffle = y.copy()
    np.random.shuffle(y_shuffle)
    
    rates = []
    preds = []
    train_preds = []
    y_true = []
    shuffle_rates = []
    class_fracs = []
    coefficients = []
    train_probs = []
    probs = []
    clfs = []
    for i,train_test_dict in enumerate(CV_folds):
        print("\t CV fold:",i,'/',len(CV_folds))

        # Get target data
        y_train = y[train_test_dict['train_idx']]
        y_test = y[train_test_dict['test_idx']]
                    
        # Make shuffled y data for empirical null
        np.random.shuffle(y_shuffle)
        y_train_shuffle = y_shuffle[train_test_dict['train_idx']]
        y_test_shuffle = y_shuffle[train_test_dict['test_idx']]

        # Variable selection
        X_train_sel = X[train_test_dict['train_idx'],:]
        X_test_sel = X[train_test_dict['test_idx'],:]
        if select_idx is not None:
            X_train_sel = X_train_sel[:,select_idx]
            X_test_sel = X_test_sel[:,select_idx]
            
        if sample_weights is not None:
            sample_weights_train = sample_weights[train_test_dict['train_idx']]
        else:
            sample_weights_train = None
                    
        # Fit model to shuffle null
        clf.fit(X_train_sel, y_train_shuffle, sample_weight=sample_weights_train)
        pred_shuffle = clf.predict(X_test_sel)
        shuffle_rates.append(np.sum(np.abs(y_test_shuffle-pred_shuffle))/len(y_test_shuffle))

        # Fit model to data 
        clf.fit(X_train_sel, y_train, sample_weight=sample_weights_train)
        pred = clf.predict(X_test_sel)
        preds.append(pred)
        train_preds.append(clf.predict(X_test_sel))
        coefficients.append(clf.coef_)
        y_true.append(y_test)
        rates.append(np.sum(np.abs(y_test-pred))/len(y_test))
        class_fracs.append(np.sum(y_train)/len(y_train))
        clfs.append(clf)
        try:
            train_probs.append(clf.predict_proba(X_train_sel))
            probs.append(clf.predict_proba(X_test_sel))
        except Exception as e:
            print(e)
                        
    out_dict = {'predictions':preds,
                'training_predictions':train_preds,
                'y_true':y_true,
                'rates':rates,
                'shuffle_rates':shuffle_rates,
                'class_fracs':class_fracs,
                'select_idx':select_idx,
                'coefficients':coefficients,
                'probabilities':probs,
                'training_probabilities':train_probs,
                'classifiers':clfs}
    return out_dict

def print_classification_results(clf_dict):
    print("Error rate/std:",np.mean(clf_dict['rates']), np.std(clf_dict['rates']))
    print("Shuffle rate/std:",np.mean(clf_dict['shuffle_rates']), np.std(clf_dict['shuffle_rates']))
    print("Class fraction:",1.0-np.mean(clf_dict['class_fracs']), np.std(clf_dict['class_fracs']))
