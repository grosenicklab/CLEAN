# Copyright (C) Responsive Capital Management - All Rights Reserved.
# Unauthorized copying of any code in this directory, via any medium is strictly prohibited.
# Proprietary and confidential.
# Written by Logan Grosenick <logan@responsive.ai>, 2019.

import numpy as np
from sklearn.preprocessing import StandardScaler

def event_regression(model, X, y, CV_folds, select_idx=None, rescale=False, sample_weights=None):
    # Rescale variables to have same norm
    if rescale:
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(X)

    # Setup variables and containers
    n = X.shape[0]
    y_shuffle = y.copy()
    np.random.shuffle(y_shuffle)
    goodness_of_fit = []
    preds = []
    train_preds = []
    y_true = []
    #shuffle_GoF = []
    #coefficients = []
    models = []

    # Iterate over CV folds
    for i,train_test_dict in enumerate(CV_folds):
        print('\t CV fold:',i,'/',len(CV_folds))

        # Get target data
        y_train = y[train_test_dict['train_thresh_idx']]
        y_test = y[train_test_dict['test_thresh_idx']]

        # Make shuffled y data for empirical null 
        #np.random.shuffle(y_shuffle)
        #y_train_shuffle = y_shuffle[train_test_dict['train_thresh_idx']]
        #y_test_shuffle = y_shuffle[train_test_dict['test_thresh_idx']]

        # Variable selection 
        X_train_sel = X[train_test_dict['train_thresh_idx'],:]
        X_test_sel = X[train_test_dict['test_thresh_idx'],:]
        if select_idx is not None:
            X_train_sel = X_train_sel[:,select_idx]
            X_test_sel = X_test_sel[:,select_idx]
        
        if sample_weights is not None:
            sample_weights_train = sample_weights[train_test_dict['train_thresh_idx']]
        else:
            sample_weights_train = None

        # Fit model to shuffle null 
        #model.fit(X_train_sel, y_train_shuffle, sample_weight=sample_weights_train)
        #pred_shuffle = model.predict(X_test_sel)
        #shuffle_GoF.append(np.sum(np.abs(y_test_shuffle-pred_shuffle))/len(y_test_shuffle))

        # Fit model to data 
        model.fit(X=X_train_sel, y=y_train, sample_weight=sample_weights_train)
        pred = model.predict(X=X_test_sel)
        preds.append(pred)
        train_preds.append(model.predict(X_test_sel))
        #coefficients.append(model.coef_)
        y_true.append(y_test)
        goodness_of_fit.append(np.sum(np.abs(y_test-pred))/len(y_test))
        models.append(model)

    out_dict = {'predictions':preds,
                'training_predictions':train_preds,
                'y_true':y_true,
                'goodness_of_fit':goodness_of_fit,
                #'shuffle_goodness_of_fit':shuffle_GoF,
                'select_idx':select_idx,
                #'coefficients':coefficients,
                'models':models}
    return out_dict

def print_magnitude_prediction(magnitude_results):
    rvals = []
    pvals = []
    std_errs = []
    for fold in range(len(magnitude_results['y_true'])):
        pred = magnitude_results['predictions'][fold]
        y_test = magnitude_results['y_true'][fold]
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(pred,y_test)
        rvals.append(r_value)
        pvals.append(p_value)
        std_errs.append(std_err)
    print('Test Prediction R-value mean/std',np.mean(rvals),np.std(rvals))
    print('Test Prediction P-value mean/std',np.mean(pvals),np.std(pvals))
    print('Test Prediction Standard Error mean/std:',np.mean(std_errs),np.std(std_errs))
    
def plot_magnitude_prediction(magnitude_results,fold=0):
    import matplotlib.pyplot as plt
    import seaborn as sns

    pred = magnitude_results['predictions'][fold]
    y_test = magnitude_results['y_true'][fold]

    plt.figure(figsize=(6,6))
    plt.scatter(pred,y_test,color='k',alpha=0.3)
    plt.xlabel("Predicted Magnitudes (log Euro)")
    plt.ylabel("Observed Magnitudes (log Euro)")
    sns.despine()

    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(pred,y_test)
    line_x = np.arange(pred.min(), pred.max())
    line_y = slope*line_x + intercept
    plt.plot(line_x, line_y,color='red',lw=3,
                 label='$R^2=%.2f$' % (r_value**2))
    plt.legend(loc='best')
    print("r-squared:", r_value**2)
    
