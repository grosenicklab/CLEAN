# Copyright (C) Responsive Capital Management - All Rights Reserved.
# Unauthorized copying of any code in this directory, via any medium is strictly prohibited.
# Proprietary and confidential.
# Written by Logan Grosenick <logan@responsive.ai>, 2019.

import numpy as np

def balance_binary_data(X,y):
    classes = np.unique(y)
    print('classes',classes)
    c0_idx = np.where(y==classes[0])[0]
    c1_idx = np.where(y==classes[1])[0]
    X0 = X[c0_idx,:]
    X1 = X[c1_idx,:]
    y0 = y[c0_idx]
    y1 = y[c1_idx]
    class_frac = float(np.sum(y))/float(y.shape[0])
    print("Class fraction:",class_frac)
    if class_frac < 0.5:
        X0 = X0[0:len(c1_idx),:]
        y0 = y0[0:len(c1_idx)]
    else:
        X1 = X1[0:len(c0_idx),:]
        y1 = y1[0:len(c0_idx)]
    X_bal = np.vstack((X0,X1))
    y_bal = np.hstack((y0,y1))
    return X_bal, y_bal

def get_CV_indices(n, y=None, train_fraction=0.6, validation_fraction=0.0, y_thresh=0.0):
    '''                                                                                                                
    Given:
        n: number of total observations to yield indices for.
        y: target variable
        train_fraction: fraction of training samples to select  
        validation_fraction: fraction of validation samples to select (0.0 yields just train/test indices)
    Returns: 
        Dict containing:  
            train_idx: training set indices     
            train_thresh_idx: magnitude training set indices corresponding to y-values above 'y_thresh'
            validate_idx: validation set indices
            test_idx: test set indices                                                                  
    '''
    out_dict = dict()
    test_fraction = 1.0 - train_fraction - validation_fraction
    train = np.random.choice(np.array(range(n)), size=int(train_fraction*n), replace=False)
    if y is not None:
        train_thresh = np.array([i for i in train if y[i] > y_thresh])
    test = np.array([i for i in range(n) if i not in train])
    if y is not None:
        test_thresh = np.array([i for i in test if y[i] > y_thresh])
    if validation_fraction > 0.0:
        validate = np.random.choice(test,size=int(validation_fraction*n),replace=False)
        test = np.array([i for i in range(n) if i not in np.hstack((train,validate))])    
        out_dict['validate_idx'] = validate
    out_dict['train_idx'] = train
    out_dict['test_idx'] = test
    if y is not None:
        out_dict['train_thresh_idx'] = train_thresh
        out_dict['test_thresh_idx'] = test_thresh
    return out_dict

def get_CV_folds(n, y, folds, train_fraction=0.6, validation_fraction=0.0):
    '''
    Generate a list of dictionaries of train/test/validation indices. 
    
    Given:  
        n: number of total observations to yield indices for
        folds: number of folds to generate
        train_fraction: fraction of training samples to select  
        validation_fraction: fraction of validation samples to select (0.0 yields just train/test indices) 
    Returns:                                                      
        List of Dicts, each containing: 
            train_idx: training set indices     
            validate_idx: validation set indices
            test_idx: test set indices 
    '''
    out_folds = []
    for f in range(folds):
        out_folds.append(get_CV_indices(n, y, train_fraction, validation_fraction))
    return out_folds
    
def k_ahead_indices(T, k_ahead=1, window=True, window_len=30):
    '''     
    Given:   
        T: total number of time points
        k_ahead: number of time points ahead in time to have in test set. 
        window: boolean. If true, use sliding widow on indices, otherwise keep all indices.
        window_len: int. if 'window' is True, window size. Otherwise, size of starting block.
    Returns:        
        window_idx: list of time indices for training
        k_ahead_idx: list of time indices for k-step-ahead prediction         
    '''
    from janus.oplib.features import rolling_window
    win_idx = rolling_window(np.array(range(T-k_ahead)),window=window_len).tolist()
    if not window:
        for i in range(len(win_idx)):
            win_idx[i] = np.array(range(win_idx[i][-1]+1))
    k_ahead_idx = []
    for i in range(len(win_idx)):
        k_ahead_idx.append(np.array(range(win_idx[i][-1]+1,win_idx[i][-1]+k_ahead+1)))
    return win_idx, k_ahead_idx



    
