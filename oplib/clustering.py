# Copyright (C) Responsive Capital Management - All Rights Reserved.
# Unauthorized copying of any code in this directory, via any medium is strictly prohibited.
# Proprietary and confidential.
# Written by Logan Grosenick <logan@responsive.ai>, 2019.

def get_centroids(X,cluster_labels):
    centroids = []
    for i,c in enumerate(np.unique(cluster_labels)):
        X_c = X[np.where(cluster_labels == c)[0],:]
        centroids.append(np.median(X_c,axis=0))
    return centroids

def get_nearest_user(point, X, cluster_labels):
    diffs = X - point
    diffs = np.sum(np.abs(diffs)**2,axis=1)
    return np.where(diffs == np.min(diffs))[0][0]
