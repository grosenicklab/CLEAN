# Copyright (C) Responsive Capital Management - All Rights Reserved.
# Unauthorized copying of any code in this directory, via any medium is strictly prohibited.
# Proprietary and confidential.
# Written by Logan Grosenick <logan@responsive.ai>, 2019.

import numpy as np
import sklearn.preprocessing as preprocessing

def simple_screen(X,y,CV_folds,num_select=20,selection=np.median):
    rank_idxs = []
    for i,fold_dict in enumerate(CV_folds):
        print('\t Fold:',i,'/',len(CV_folds))
        # For binary event detection, use full data, otherwise use thresholded y
        # for magnitude prediction. 
        if len(np.unique(y)) == 2:
            X_train = X[fold_dict['train_idx'],:]
            y_train = y[fold_dict['train_idx']]
        else:
            X_train = X[fold_dict['train_thresh_idx']]
            y_train = y[fold_dict['train_thresh_idx']]
        rank_idx, _ = med_corr_select(X_train,y_train,num_select=num_select)
        rank_idxs.append(rank_idx)
    select_ranks = np.apply_along_axis(selection,0,np.asarray(rank_idxs))
    return select_ranks, np.argsort(select_ranks)[::-1][0:num_select]

def get_corrs(X,y):
    '''
    Return Pearson correlations of y with each column of X.
    '''
    n = y.shape[0]
    y.shape = (n,1)
    scaler = preprocessing.StandardScaler()
    y_s = scaler.fit_transform(y)
    X_s = scaler.fit_transform(X)
    corrs = np.dot(X_s.T,y_s)/float(n)
    return corrs

def get_tvals(X,y,denom_reg=1e-10):
    '''
    Return t-statistics (unequal variance) for each column of X grouped by binary outcomes in y.
    '''
    yvals = np.unique(y)
    c1_idx = np.where(y==yvals[0])[0]
    c2_idx = np.where(y==yvals[1])[0]
    n1 = float(len(c1_idx))
    n2 = float(len(c2_idx))
    X1 = X[c1_idx,:]
    X2 = X[c2_idx,:]
    X1_bar = np.mean(X1,axis=0)
    X2_bar = np.mean(X2,axis=0)
    v1 = np.nan_to_num(np.std(X1,axis=0)**2)
    v2 = np.nan_to_num(np.std(X2,axis=0)**2)
    num = X2_bar - X1_bar
    if n1 > 0 and n2 > 0:
        denom = np.nan_to_num(np.sqrt(v1/n1 + v2/n2)) + denom_reg
        return num / denom
    else:
        return 0.0

def resample_corrs(X,y,n_reps=100,sample_frac=0.6,replace=False,return_corrs=False,shuffle=False,verbose=False):
    '''
    Return Pearson median correlations of y with each column of X over 'n_reps' replicates, resampled 
    at resampling fraction 'sample_frac' with or without replacement. If y is a binary variable, return 
    unequal variance t-statistics for the groups defined by the classes in y.
    '''
    n = len(y)
    if len(np.unique(y)) <= 2:
        binary = True
        if verbose:
            print("\t Using binary tval approach.")
    else:
        binary = False
    corrs = []
    y_var = y.copy()
    for r in range(n_reps):
        if shuffle:
            np.random.shuffle(y_var)
        tr_idx = np.random.choice(np.array(range(n)), size=int(sample_frac*n), replace=replace)
        ts_idx = np.array([i for i in range(n) if i not in tr_idx])
        X_rep = X[tr_idx,:]
        y_rep = y_var[tr_idx]
        if binary:
            corrs.append(get_tvals(X_rep,y_rep))
        else:
            corrs.append(get_corrs(X_rep,y_rep))
    
    if return_corrs:
        return np.squeeze(np.asarray(corrs))
    else:
        return np.median(np.squeeze(np.nan_to_num(np.asarray(corrs))),axis=0)
    
def resample_corr_imap(in_args):
    idx, y, X, binary, shuffle, sample_frac, replace = in_args
    n = len(y)
    y_var = y.copy()
    if shuffle:
        np.random.shuffle(y_var)
    tr_idx = np.random.choice(np.array(range(n)), size=int(sample_frac*n), replace=replace)
    ts_idx = np.array([i for i in range(n) if i not in tr_idx])
    X_rep = X[tr_idx,:]
    y_rep = y_var[tr_idx]
    if binary:
        corrs = get_tvals(X_rep,y_rep)
    else:
        corrs = get_corrs(X_rep,y_rep)
    return (idx,corrs)

def med_corr_select(X_train,y_train,num_select=20):
    median_corrs = resample_corrs(X_train,y_train)
    rank_idx = np.argsort(np.abs(median_corrs))[::-1]
    select_idx = rank_idx[range(num_select)]
    return rank_idx, select_idx
