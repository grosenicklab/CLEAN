# Copyright (C) Responsive Capital Management - All Rights Reserved.
# Unauthorized copying of any code in this directory, via any medium is strictly prohibited.
# Proprietary and confidential.
# Written by Logan Grosenick <logan@responsive.ai>, 2019.

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.distance import pdist, squareform

from .dataio import time_normalize
from .features import cross_corr

font = {'family' : 'normal',
        'weight' : 'regular',
        'size'   : 20}
plt.rc('font', **font)

def plot_user(data,user_id):
    '''
    Plot slice of data_tensor for user 'user_id' as an image.
    '''
    plt.figure(figsize=(15,5))
    ax = plt.gca()
    user_dat = time_normalize(data.tensor[user_id,:,:])
    im_min = np.min(user_dat.T)
    im_max = np.max(user_dat.T)
    plt.imshow(user_dat.T,aspect='auto',cmap=cm.viridis,vmin=im_min,vmax=im_max)
    plt.xlabel("Time (Weeks)")
    plt.ylabel("Features")
    return ax
    
def plot_user_feature(data,user_id,feature_id):
    '''
    Plot time series of data_tensor for user 'user_id' and feature 'feature_id'.
    '''
    plt.figure(figsize=(15,5))
    ax = plt.gca()
    plt.plot(np.abs(data.tensor[user_id,:,feature_id]),color='k')
    xtickNames = data.times_unique 
    plt.xticks(np.array(range(len(xtickNames)))[0::10], xtickNames[0::10],rotation=60)
    plt.ylabel('Euro')
    plt.title(data.feature_LUT[data.features_unique[feature_id]])
    sns.despine()
    return ax

def plot_recurrance(data,user_id,feature_id):
    def rec_plot(s, eps=0.10, steps=10):
        d = pdist(s[:,None])
        d = np.floor(d/eps)
        d[d>steps] = steps
        Z = squareform(d)
        return Z

    user_dat = time_normalize(data.tensor[user_id,:,:],norm_type='mean')

    plt.figure(figsize=(10,10))
    rec = rec_plot(user_dat[:,feature_id])
    plt.imshow(rec) #,cmap=cm.viridis)
    plt.title("Recurrance plot for "+data.feature_LUT[data.features_unique[feature_id]])
    plt.xlabel('Time (weeks)')
    plt.ylabel('Time (weeks)')
    plt.show()
    
def plot_feature(data,feature_id,data_oos=None,smoothing=None):
    '''
    Plot a users by time image for feature given by 'feature_num'. Note this is a direct index
    into the feature LUT (a 'feature number', not a feature id corresponding to a data tensor index.
    e.g., feature_num = 270099780 corresponding to feature_LUT {270099780: 'cf_transfer.investment'}.

    '''
    from scipy import signal
    from sklearn.preprocessing import StandardScaler
    feature_dat = np.squeeze(data.tensor[:,:,feature_id])
    if data_oos is not None:
        feature_dat_oos = np.squeeze(data_oos.tensor[:,:,feature_id])

    bound = np.max((np.abs(np.min(feature_dat)),np.max(feature_dat)))
    print("Min/max for",str(data.feature_LUT[data.features_unique[feature_id]]),":",np.min(feature_dat),np.max(feature_dat))

    # For transfers, all values are negative, so use sequential colormap from zero
    im = np.abs(feature_dat)

    # Sort by something that gives visual coherence
    #scaler = StandardScaler()
    im_scaled = im.T #scaler.fit_transform(im.T)

    im_tnorm = np.sqrt(im_scaled.T) # time_normalize(-1*im_scaled.T,t_dim=0)
    im_min = np.min(im_tnorm)
    im_max = np.max(im_tnorm)
    if smoothing is not None:
        im_tnorm_sm = np.apply_along_axis(signal.savgol_filter,1,im_tnorm,window_length=smoothing,polyorder=3)
    else: 
        im_tnorm_sm = im_tnorm 
        
    from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
    Z = linkage(im_tnorm_sm, 'ward')
    ids = leaves_list(Z)
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import pdist
    c, coph_dists = cophenet(Z, pdist(im_scaled.T))
    print("Cophenet:",c)

    im_ord = im_tnorm[ids,:]
    nonzero_idx = np.where(np.abs(np.sum(im_ord,axis=1)) > 0.0)[0]

    plt.figure(figsize=(12,12))
    plt.title("Time intensity plot for "+data.feature_LUT[data.features_unique[feature_id]])
    print("Time intensity plot for "+data.feature_LUT[data.features_unique[feature_id]])
    grid = plt.GridSpec(1, 100, wspace=0.3) 
    
    if data_oos is None:
        plt.imshow(im_ord[nonzero_idx,:],aspect='auto',cmap=cm.inferno,vmin=0.01*im_min,vmax=0.01*im_max)
    else:
        plot_ratio = int(100*feature_dat.shape[1]/(feature_dat_oos.shape[1]+feature_dat.shape[1]))
        plt.subplot(grid[0, 0:plot_ratio])
        plt.imshow(im_ord[nonzero_idx,:],aspect='auto',cmap=cm.inferno,vmin=0.01*im_min,vmax=0.01*im_max)
        plt.xlabel("Time (weeks)")
        plt.ylabel("User ID")

        plt.subplot(grid[0, (plot_ratio+1):])
        im_oos = np.abs(feature_dat_oos)
        im_oos_tnorm = np.sqrt(im_oos)
        im_ord_oos = im_oos_tnorm[ids,:]
        plt.imshow(im_ord_oos[nonzero_idx,:],aspect='auto',cmap=cm.inferno,vmin=0.01*im_min,vmax=0.01*im_max)

    plt.tight_layout()        
    cbar = plt.colorbar()
    cbar.set_label('SQRT(Euro)', rotation=270,labelpad=30)
    sns.despine()

    plt.figure(figsize=(12,6))
    grid = plt.GridSpec(1, 100, wspace=0.3)
    if data_oos is None:
        plt.plot(np.sum(-1*feature_dat,axis=0))
        plt.xlabel("Time (weeks)")
        plt.ylabel("Total Euro")
    else:
        plot_max = np.max((np.max(-1.*feature_dat),np.max(-1.*feature_dat_oos)))
        plt.subplot(grid[0, 0:plot_ratio])
        plt.plot(np.sum(-1*feature_dat,axis=0))
        plt.ylim([0,plot_max])
        plt.xlabel("Time (weeks)")
        plt.ylabel("Total Euro")
        plt.subplot(grid[0, (plot_ratio+1):])
        plt.plot(np.sum(-1*feature_dat_oos, axis=0))
        plt.ylim([0,plot_max])
        frame1 = plt.gca()
        frame1.axes.get_yaxis().set_visible(False)
    plt.tight_layout()        
    
def plot_xcor(data,user_id):
    user_data = data.tensor[user_id,:,:]
    med_max_xcor = cross_corr(user_data)
    plt.figure(figsize=(8,8))
    im_min = np.min(med_max_xcor)
    im_max = np.max(med_max_xcor)
    bound = np.max((np.abs(im_min),np.abs(im_max)))
    plt.imshow(np.abs(med_max_xcor),aspect='auto',cmap=cm.viridis,vmax=10000)
    plt.title("Cross-correlation plot")
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.show()
    
def plot_regression_scatter(pred,y_true):
    plt.figure(figsize=(6,6))
    plt.scatter(pred,y_true,color='k',alpha=0.3)
    plt.xlabel("Predicted Magnitudes (log Euro)")
    plt.ylabel("Observed Magnitudes (log Euro)")
    sns.despine()

    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(pred,y_test_n)
    line_x = np.arange(pred.min(), 2.0)
    line_y = slope*line_x + intercept
    plt.plot(line_x, line_y,color='red',lw=3, label='$R^2=%.2f$' % (r_value**2))
    plt.legend(loc='best')
    print("r-squared:", r_value**2)
