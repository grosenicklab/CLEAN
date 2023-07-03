# Copyright (C) Responsive Capital Management - All Rights Reserved.
# Unauthorized copying of any code in this directory, via any medium is strictly prohibited.
# Proprietary and confidential.
# Written by Logan Grosenick <logan@responsive.ai>, 2019.

# Feature notes: at the moment we are rolling up features weekly. Note that if we ever go sub-weekly, for OP transactions from the weekend all pile up on Monday (no transactions come through Saturday, Sunday). Further, as a rule-of-thumb there should be roughly as many trasactions daily as there are users (across all transactions). 

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

def MAD(vec):
    '''
    Median absolute deviation of vector 'vec'.
    '''
    med = np.median(vec)
    return np.median(np.abs(vec-med))

def get_events(vec,thresh='MAD',coef=1.4826):
    '''
    Given a thresholding mechanism, get indices for events in a vector.
    '''
    if thresh is 'MAD':
        mad = MAD(vec)
        events = np.where(np.abs(vec)>coef*mad)[0]
    else: 
        events = np.where(np.abs(vec)>0.0)[0]
    return events

def interevent(vec,time_base=None,min_events=1):
    '''
    Return interevent distribution for vector 'vec' and threshold thresh.
    '''
    events = get_events(vec,thresh='zero') #thresh='MAD',coef=1.4826)
    if time_base:
        diffs =  time_base*np.diff(events)
        if len(diffs) < min_events:
            return np.nan, np.nan
        else:
            return np.median(diffs), MAD(diffs)
    else:
        diffs =  np.diff(events)
        if len(diffs) < min_events:
            return np.nan, np.nan
        else:
            return np.median(diffs), MAD(diffs)
        
def last_event(vec,time_base=None):
    '''
    Return interevent distribution for vector 'vec' and threshold thresh.
    '''
    n = len(vec)
    events = get_events(vec,thresh="zero") #,thresh='MAD',coef=1.4826)
    if len(events) > 0:
        last_idx = n - events[0]
    else:
        last_idx = np.nan
    if time_base is not None:
        last_idx *= time_base
    return last_idx
        
def perievent_hist(vec,event_idx,hist_width=10):
    '''
    Return array of pre-event histograms for given event_idxs and width.
    '''
    hist_vals = []
    for t in event_idx:
        if t > hist_width:
            vals = vec[(t-hist_width):t]
            hist_vals.append(vals)
        else:
            vals = np.zeros(hist_width)
            vals[(hist_width-t):] = vec[0:t]
            hist_vals.append(vals)
    return np.asarray(hist_vals)  

def cross_corr(X,feature_names=None):
    '''
    Return max and median of cross correlation between columns of X.
    If a feature names array is provided, an array of feature names for each xcor is returned.
    '''
    med_max_xcor = np.zeros((X.shape[1],X.shape[1]))
    feature_mat = np.empty((X.shape[1],X.shape[1])).astype(str)
    for i in range(X.shape[1]):
        if feature_names is not None:
            feature_mat[i,i] = str(feature_names[i])+'_'+str(feature_names[i])
        for j in range(X.shape[1]):
            if i < j:
                xcor = np.correlate(X[:,i],X[:,j])
                med_max_xcor[i,j] = np.max(xcor)
                med_max_xcor[j,i] = np.median(xcor)
                if feature_names is not None:
                    feature_mat[i,j] = str(feature_names[i])+'_'+str(feature_names[j])
                    feature_mat[j,i] = str(feature_names[j])+'_'+str(feature_names[i])
    if feature_names is not None:
        return med_max_xcor, feature_mat
    else:
        return med_max_xcor

def outlier_score(data_mat,contamination_thresh=None,novelty=False):
    '''
    Assign outlier scores to each multivariate column of data_mat.
    '''
    np.random.seed(42)
    if contamination_thresh is None:
        clf = LocalOutlierFactor(n_neighbors=20, contamination="auto",novelty=novelty)
    else: 
        clf = LocalOutlierFactor(n_neighbors=20, contamination=contamination_thresh,novelty=novelty)
    y_pred = clf.fit_predict(data_mat)
    return clf.negative_outlier_factor_

def get_stats_matrix(user_data, feature_names=None, xcorr=True):
    '''
    Given a features x time matrix for a particular user 'user_data', generate an output matrix of statistics over time
    Args:
        user_data: slice of taxonomy data tensor for a single user, possibly windowed in time.
        features_unique: array of feature numbers (found in data tensor object as data.features_unique)
        features: If None, returns just stats matrix for user. If a list of feature numbers, returns stats matrix
                  and features names for the given features.
    Returns:
        stats: an features x stats array 
        out_vars: variable labels giving stat and variable name, e.g., 'Max_IEI_270109591'. 
        
    '''
    if feature_names is not None:
        out_vars = []
    # Inter-event interval median and MAD (on event times)
    iei = np.apply_along_axis(interevent,0,user_data)
    median_iei = iei[0,:]
    MAD_iei = iei[1,:]
    stats = np.array(median_iei)
    stats = np.vstack((stats,MAD_iei))
  
    # Last event
    last = np.apply_along_axis(last_event,0,user_data)
    stats = np.vstack((stats,last))

    # Fano factor (on event times)
    fano = MAD_iei/median_iei
    stats = np.vstack((stats,fano))

    # median and max event magnitude (on event intensity)
    median_val = np.apply_along_axis(np.median,0,user_data)
    max_val = np.apply_along_axis(np.max,0,user_data)
    stats = np.vstack((stats, median_val))
    stats = np.vstack((stats, max_val))

    # Number of outlying/novely events
    oscores = outlier_score(user_data.T)
    stats = np.vstack((stats,oscores))

    if xcorr:
        # Max cross-correlation 
        if feature_names is not None:
            med_max_xcor, feature_mat = cross_corr(user_data,feature_names)
        else:
            med_max_xcor = cross_corr(user_data,feature_names)
    
        stats = np.vstack((stats, med_max_xcor))
        
    if feature_names is not None:
        out_vars.extend(['Median_IEI_'+str(feature_names[i]) for i in range(len(median_iei))])
        out_vars.extend(['MAD_IEI_'+str(feature_names[i]) for i in range(len(MAD_iei))])
        out_vars.extend(['Last_Event_'+str(feature_names[i]) for i in range(len(last))])
        out_vars.extend(['Fano_'+str(feature_names[i]) for i in range(len(fano))])
        out_vars.extend(['Median_Magnitude_'+str(feature_names[i]) for i in range(len(median_val))])
        out_vars.extend(['Max_IEI_'+str(feature_names[i]) for i in range(len(max_val))])
        out_vars.extend(['Outlier_Score_'+str(feature_names[i]) for i in range(len(oscores))])
        if xcorr:
            out_vars.extend(['Max_Xcor_'+str(feature_mat.flatten()[i]) for i in range(len(feature_mat.flatten()))])
            out_vars.extend(['Median_Xcor_'+str(feature_mat.flatten()[i]) for i in range(len(feature_mat.flatten()))])
        return stats, out_vars
    else:
        return stats

def interevent_fano(arr):
    '''
    Get interevent Fano factor for each of provided array.
    '''
    # Inter-event interval median and MAD (on event times)
    iei = np.apply_along_axis(interevent,0,arr)
    median_iei = iei[0,:]
    MAD_iei = iei[1,:]

    # Fano factor (on event times)
    fano = MAD_iei/median_iei
    return fano

from multiprocessing import Pool

def rolling_window(a, window):
    '''
    Apply a rolling window to vector a of size 'window'. 
    '''
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def binarize(vec,thresh=0.0):
    '''
    Binarize values in vec with 0s below thresh and 1s above.
    '''
    vec_out = np.zeros_like(vec)
    nonzero = np.where(np.abs(vec)>thresh)[0]
    vec_out[nonzero] = 1.0
    return vec_out    

def rolling_arr_window(arr,window,pad_start=None,pad_end=None):
    ''' 
    Return a rolling window applied to the columns of an array. 
    Args:
        arr: the numpy array to be windowed.
        window: integer. the window size.
        pad: boolean. If true, zeros will be appended to the start so that the
             number of resulting windows matches the number of rows in arr.
    '''
    if pad_start is not None:
        pad = np.zeros((pad_start,arr.shape[1]))
        arr = np.vstack((pad,arr))
    if pad_end is not None:
        pad = np.zeros((pad_end,arr.shape[1]))
        arr = np.vstack((arr,pad))
    return np.apply_along_axis(rolling_window,0,arr,window=window)

def windowed_features(user_arr,window_len=10,k_ahead=1,stats=True,xcorr=True,crop=True):
    '''                   
    Given a user data array (time, features), and a feature id to use as the prediction variable, 
    yield data matrices for 1-step ahead prediction of the feature using the other variables in a window of length 
    'win_length'. 
    '''
    # Get windowed data in list, padded to allow windowing and k-step ahead prediction    
    n,p = user_arr.shape
    user_data = rolling_arr_window(user_arr,window_len+k_ahead,pad_start=window_len-1,pad_end=k_ahead)

    # Initialize containers and loop over windowed data list to calculate stats
    data_win = []
    target_win = []
    stats_win = []
    for win_data in user_data:
        # TODO: add Parzen window weights 
        data_win.append(win_data[:-k_ahead,:])
        target_win.append(win_data[window_len:,:])
        if stats:
            # Note inter-event nans are currently being converted to zero  
            stats_win.append(np.nan_to_num(get_stats_matrix(data_win[-1],xcorr=xcorr)))
    if stats:
        win_features = np.asarray(stats_win)
    else:
        win_features = np.asarray(data_win)
        #win_features = win_features.reshape((win_features.shape[0],np.prod(win_features.shape[1:3])))
    if crop:
        return win_features[(window_len-1):-k_ahead,:,:], np.asarray(target_win)[(window_len-1):-k_ahead,:,:]
    else:
        return win_features, np.asarray(target_win)

def windowed_features_imap(in_tuple):
    '''          
    Multiprocessing imap wrapper for feature_predict. 
    '''
    idx, user_mat, win_len, k_ahead, xcorr, stats = in_tuple
    X,Y = windowed_features(user_mat,window_len=win_len,xcorr=xcorr,stats=stats)
    return [idx,X,Y]

def get_windowed_features(data,n_users=2000,window_len=10,k_ahead=1,xcorr=True,stats=True,savepath=None):
    '''     
    Multithreaded function to get windowed features for prediction. 
    '''
    in_args = [(i,data.tensor[i,:,:],window_len,k_ahead,xcorr,stats) for i in range(n_users)]

    p = Pool(64)
    results = p.imap(windowed_features_imap,in_args,chunksize=10)

    X = []
    Y = []
    print("Generating windowed features...")
    for i,r in enumerate(results):
        if np.mod(i,10) == 0:
            print('\t...',r[0])
        X_u = r[1]
        Y_u = r[2]
        X.append(X_u)
        Y.append(Y_u)
    if savepath is not None:
        print('Saving windowed features to',savepath)
        np.savez(savepath,windowed_features=np.asarray(X),target_features=np.asarray(Y))
    return X, Y

def feature_QC(data,sigma=3.5,window_size=10):
    '''
    Quality control for features, to check if something in the feature pipeline might be broken, 
    and more generally to alert us if there are large changes in feature value across clients.
    '''
    import matplotlib.pyplot as plt
    outliers = []
    for feature_num in data.features_unique:
        # get the feature sum across users over time (windowed by length window_size)
        f_idx = np.where(data.features_unique==feature_num)[0]
        feature_dat = np.squeeze(data.tensor[:,:,f_idx])
        feature_sum = np.sum(np.abs(feature_dat),axis=0) # sum abs value of feature values over users 
        feature_win_list = rolling_window(feature_sum,window=window_size) # get time windowed data
        feat_win = np.mean(np.asarray(feature_win_list),axis=1) # time windowed mean
        # Get the global mean and standard deviation of the windowed time average
        feat_win_mean = np.mean(feat_win) # global mean
        feat_win_std = np.std(feat_win) # 
        # Detect     
        below = feat_win - feat_win_mean < -1*sigma*feat_win_std
        above = feat_win - feat_win_mean > sigma*feat_win_std
        if np.any(below):
            print('\t ',data.feature_LUT[feature_num],'is <',str(sigma),'sigma below a',str(window_size)+
                 ' moving average for', str(len(np.where(below)[0])), 'weeks.' )
            plt.figure(figsize=(15,1))
            plt.plot(feat_win)
            plt.title(data.feature_LUT[feature_num])
            plt.xlabel('Time (weeks)')
            plt.ylabel('Rolling mean (Euro)')
            outliers.append(feature_sum)
        if np.any(above):
            print('\t ',data.feature_LUT[feature_num],'is >',str(sigma),'sigma below a',str(window_size)+
                 ' moving average for', str(len(np.where(above)[0])), 'weeks.' )
            plt.figure(figsize=(15,1))
            plt.plot(feat_win)
            plt.title(data.feature_LUT[feature_num])
            plt.xlabel('Time (weeks)')
            plt.ylabel('Rolling mean (Euro)')
            outliers.append(feature_sum)
    plt.figure(figsize=(10,10))    
    import seaborn as sns
    corr = np.corrcoef(np.asarray(outliers))
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    ax = sns.heatmap(corr, mask=mask, vmin=-0.3, vmax=.3, square=True, cmap="bwr")
    plt.title('Correlation matrix for outliers listed above (in order listed)')
    
