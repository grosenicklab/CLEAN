# Copyright (C) Responsive Capital Management - All Rights Reserved.
# Unauthorized copying of any code in this directory, via any medium is strictly prohibited.
# Proprietary and confidential.
# Written by Logan Grosenick <logan@responsive.ai>, 2019.

import psycopg2 as pg
import numpy as np
import os

from dotenv import load_dotenv
load_dotenv()

PG_PORT = os.getenv('PG_PORT') or '3308'
PG_DATABASE = os.getenv('PG_DATABASE') or 'symbiosis'
PG_USER = os.getenv('PG_USER') or 'symbiosis'
PG_PASSWORD = os.getenv('PG_PASSWORD') or 'symbiosis'

def get_cursor(local_postgres_host,
                        local_postgres_port=PG_PORT, 
                        local_postgres_db=PG_DATABASE, 
                        local_postgres_user=PG_USER,
                        local_postgres_password=PG_PASSWORD):
    '''
    Get cursor for querying taxonomy database:
    Args:
        local_postgres_host: the host address, e.g., 'xxx.eu-central-1.rds.amazonaws.com'
        local_postgres_port: the local port, e.g., '5432'
        local_postgres_db: the database to use, typically 'symbiosis'
        local_postgres_user/local_postgres_password: username and password 
    Returns:
        A cursor object for local taxonomy database.
    '''
    # Connect with psycopg2
    con = pg.connect(
            host=local_postgres_host,
            port=local_postgres_port,
            user=local_postgres_user,
            password=local_postgres_password,
            dbname=local_postgres_db )
    cursor = con.cursor()
    return cursor

def load_taxonomy_data(cursor, time_range=None, features=None, clients=None, sample_max=None,
                       randomize_clients=True, time_window=0, verbose=False):
    '''                                                                                                               
    Load taxonomy data set from postgres database to numpy array.
    Args:
        cursor: the taxonomy database cursor returned by get_taxonomy_cursor. 
        time_range: if None, the full data set for all time points is returned.                                       
                  if a two-value list specifying a integer range, e.g., [0,10], the data corresponding to  these 
                      indices are returned.
                  if a two-value list specifying datetime range, e.g., ['2016-12-31','2017-01-07'], the data for 
                      that time range are returned.
                  if a two-value list ['before',<datetime>], just the data prior to the given datetime are returned.
                  if a two-value list ['since',<datetime>], just the data including and following the given datetime 
                      are returned. 
        features: if None, all features are returned, 
                  otherwise a list of integer values specifying featured ids in database to return.
        clients: if None all clients are returned; 
                 if an integer, that integer number of clients are returned; 
                 if a list of client unique ids, these specific clients are returned.
        sample_max: if None all samples are returned, 
                    if an integer, that number of samples are return from beginning of database (for debugging).
        randomize_clients: boolean. If true, when number of clients is given, they are chosen at random.

    Returns: 
        A numpy data array with rows like (user_id, transaction_date, feature_id, value):
            array(['312a1e20-caf1-4655-83df-74c8dca3745a', '2017-01-07', '84034', '15166.7'], dtype='<U36')
    '''
    # Fetch taxonomy data points  
    postgreSQL_select_Query = "select * from taxonomy_data_points"

    # ...for provided time range, defaulting to all time points
    cursor.execute("select distinct transaction_date from taxonomy_data_points")
    data_fetch = cursor.fetchall()
    unique_dates = np.sort(np.asarray(data_fetch).flatten())
    if time_range is not None:
        if type(time_range[0]) is str:
            if time_range[0] is 'before':
                date1 = unique_dates[0]
                date2 = unique_dates[np.where(unique_dates==time_range[1])[0][0] - 1]
            elif time_range[0] is 'since':
                date1 = unique_dates[np.where(unique_dates==time_range[1])[0][0] - time_window]
                date2 = unique_dates[-1]
            else:
                date1 = time_range[0]
                date2 = time_range[1]
        elif type(time_range[0]) is int:
            date1 = unique_dates[np.where(unique_dates==time_range[0])[0][0] - time_window]
            date2 = unique_dates[time_range[1]]
    else:
        date1 = unique_dates[0]
        date2 = unique_dates[-1]
    if verbose:
        print('\t Loading from:',date1,'to',date2)
    postgreSQL_select_Query += " where transaction_date between '" + str(date1) +"' and '" + str(date2) + "'"

    # ...and for provided client number or user ids:
    if clients is not None:
        cursor.execute("select distinct user_id from taxonomy_data_points")
        data_fetch = cursor.fetchall()
        unique_uids = np.sort(np.asarray(data_fetch).flatten())
        if type(clients) is int:
            if randomize_clients:
                uids = np.random.choice(unique_uids,size=clients,replace=False)
                if verbose:
                    print('\t Randomized on',str(clients),'clients.')
            else:
                uids = unique_uids[0:clients]
                if verbose:
                    print('\t On the first',str(clients),'clients.')
        elif len(clients) > 1 and type(clients[0]) is str:
            uids = clients
            if verbose:
                print('\t On the specified list of',str(len(clients)),'clients.')
        else:
            raise ValueError("Provided type/length of 'clients' not recognized." +
                             "Please provide a number of clients or a list of client user ids as strings.")
        postgreSQL_select_Query += " AND user_id in ("
        for i,u in enumerate(uids):
            if i+1 < len(uids):
                postgreSQL_select_Query += "'"+u+"',"
            else:
                postgreSQL_select_Query += "'"+u+"')"

    # ...and for specified feature ids
    if features is not None:
        if verbose:
            print('\t On',str(len(features)),'specified features.')
        cursor.execute("select distinct feature_id from taxonomy_data_points")
        data_fetch = cursor.fetchall()
        unique_features = np.sort(np.asarray(data_fetch).flatten())
        if type(features) == type(np.array([])):
            features = features.tolist()
        if(not all(f in unique_features.tolist() for f in features)): 
            raise ValueError("Provided feature list or array contains values that are not in database.")
        
        postgreSQL_select_Query += " AND feature_id in ("
        for i,f in enumerate(features):
            if i+1 < len(features):
                postgreSQL_select_Query += "'"+str(f)+"',"
            else:
                postgreSQL_select_Query += "'"+str(f)+"')"
    else:
        if verbose:
            print('\t On all features.')    
        cursor.execute("select distinct feature_id from taxonomy_data_points where transaction_date between '" +
                   str(date2) +"' and '" + str(date2) + "'")
        data_fetch = cursor.fetchall()
        unique_features = np.sort(np.asarray(data_fetch).flatten()).tolist()
        postgreSQL_select_Query += " AND feature_id in ("
        for i,f in enumerate(unique_features):
            if i+1 < len(unique_features):
                postgreSQL_select_Query += "'"+str(f)+"',"
            else:
                postgreSQL_select_Query += "'"+str(f)+"')"

    # ...possibly limited to a fixed number of data points (for debugging)
    if sample_max is not None:
        postgreSQL_select_Query += " LIMIT " + str(sample_max)

    # Execute the query
    if verbose:
        print('Postgres query:',postgreSQL_select_Query)
    cursor.execute(postgreSQL_select_Query)
    data_fetch = cursor.fetchall()

    # Move data from rows to numpy array and return it
    data = []
    for row in data_fetch:
        data.append(row)
    return np.asarray(data)

def get_feature_LUT(cursor):
    '''
    Generate feature look up table (LUT) from postgres database to dictionary.
    Args:
        cursor: the taxonomy database cursor returned by get_taxonomy_cursor.
    Returns:
        A dictionary with entries
    {... <feature number>:<feature name> ...} e.g {... 270106525: 'cf_transfer.payment.loan' ...}
            array(['312a1e20-caf1-4655-83df-74c8dca3745a', '2017-01-07', '84034', '15166.7'], dtype='<U36')
    '''
    # Query database for features
    postgreSQL_select_Query = "select * from features"
    cursor.execute(postgreSQL_select_Query)
    feature_fetch = cursor.fetchall() 
    # Organize features into LUT
    feature_LUT = dict()
    for row in feature_fetch:
        feature_LUT[row[0]] = row[1]
    return feature_LUT

class Data:
    '''
    Object storing the loaded data as a tensor. 

    Init Args:
        data: data array returned by 'load_taxonomy_data'.
        feature_LUT: feature look-up-table returned by 'get_feature_LUT'.

    Attributes:
        feature_LUT: the feature look-up-table traslating feature_ids to feature names
        users_unique: unique user ids in data set
        times_unique: unique dates included in data set
        features_unique: unique feature ids included in data set
        tensor: tha data tensor, indexed as (users,time,features)

    Methods:
        forcast_data: holds out last k time points for testing forecasting purposes.
        save: save data tensor, index values, and feature LUT to npz file. 
        load: reload data from npz file created using 'save'.
    '''
    def __init__(self,data,feature_LUT,features_unique=None,users_unique=None,num_clients=None,
                 feature_range=[270000000,2**64],verbose=True):
        
        def _map_to_idx(unique_vals):
            out_dict = dict()
            for i,val in enumerate(unique_vals):
                out_dict[val] = i
            return out_dict
        self.feature_LUT = feature_LUT
        
        self.users = np.asarray([data[i][0] for i in range(len(data))])
        if users_unique is None:
            self.users_unique = np.unique(self.users)
        else:
            self.users_unique = users_unique
        users_idx_map = _map_to_idx(self.users_unique)

        self.times = np.asarray([np.datetime64(data[i][1]) for i in range(len(data))])
        self.times_unique = np.unique(self.times)
        times_idx_map = _map_to_idx(self.times_unique)

        self.features = np.asarray([data[i][2] for i in range(len(data)) if int(data[i][2]) >= feature_range[0] and
                                    int(data[i][2]) <= feature_range[1]])
        if features_unique is None:
            self.features_unique = np.unique(self.features).astype(int)
        else:
            self.features_unique = features_unique
        features_idx_map = _map_to_idx(self.features_unique)

        self.vals = np.asarray([data[i][3] for i in range(len(data))])
        self.vals_unique = np.unique(self.vals)

        self.tensor = np.zeros(shape=(len(self.users_unique),len(self.times_unique),len(self.features_unique)))
        for i in range(len(data)):
            if int(data[i][2]) >= feature_range[0] and int(data[i][2]) <= feature_range[1]:
                u_idx = users_idx_map[data[i][0]]
                t_idx = times_idx_map[np.datetime64(data[i][1])]
                f_idx = features_idx_map[int(data[i][2])]
                self.tensor[u_idx,t_idx,f_idx] = data[i][3]
        if verbose:
            print("\t Number of nonzeros/total:",len(np.nonzero(self.tensor.flatten())[0]),"/",
                  np.prod(self.tensor.shape))
            print("\t Fraction of nonzeros/total:",len(np.nonzero(self.tensor.flatten())[0])/np.prod(self.tensor.shape))
        if num_clients is not None:
            self.tensor = self.tensor[0:num_clients,:,:]
        
    def get_first_dates(self):
        '''
        Get and return earliest dates for each user.
        '''
        first_dates = []
        for uid in self.users_unique:
            first_dates.append(min(self.times[np.where(self.users==uid)[0]]))
        self.first_dates = np.array(first_dates)
        first_counts = []
        for tid in self.times_unique:
            first_counts.append(len(np.where(self.first_dates==tid)[0]))
        self.first_counts = np.array(first_counts)
        
    def forecast_data(self,k=1):
        '''
        Return two data tensors, split over time such that the last k time points are totally held out
        as a separate tensor for out-of-sample prediction.
        '''
        n = len(self.times_unique)
        tensor_in = self.tensor[:,0:(n-k),:]
        tensor_out = self.tensor[:,-k:,:]
        return tensor_in, tensor_out

    def save(self,savepath='/tmp/data_tensor.npz'):
        '''
        Save loaded data to an npz file located at 'savepath'.
        '''
        np.savez(savepath,tensor=self.tensor,
                 users_unique=self.users_unique,
                 times_unique=self.times_unique,
                 features_unique=self.features_unique,
                 feature_LUT=self.feature_LUT)

        print('Saved tensor data to:',savepath)
        
    def load(self,savepath='/tmp/data_tensor.npz'):
        '''
        Load data previously saved at 'savepath' into Data object.
        '''
        npz_data = np.load(savepath)
        self.tensor = npz_data['tensor']
        self.users_unique = npz_data['users_unique']
        self.times_unique = npz_data['times_unique']
        self.features_unique = npz_data['features_unique']
        self.feature_LUT = npz_data['feature_LUT']

        print('Loadad data from:',savepath)
        
def time_normalize(mat,t_dim=0,thresh=1e-10,norm_type='max'):
    '''
    Normalize matrix of data so each time series has unit norm.
    Args:
        mat: numpy array
        t_dim: which axis is the time axis (0 or 1)
        thresh: if norm of time series is below thresh, series is ignored.
    Returns:
        norm_mat: time normalized array
    '''
    means = np.mean(mat,axis=t_dim)
    sds = np.std(mat,axis=t_dim)
    norm_mat = np.abs(mat.copy())
    for j in range(mat.shape[1]):
        if np.sum(np.abs(norm_mat[:,j])) > thresh:
            if norm_type is 'max':
                norm_mat[:,j] = norm_mat[:,j] / np.max(norm_mat[:,j])
            else:
                norm_mat[:,j] = norm_mat[:,j] - means[j]
                norm_mat[:,j] = norm_mat[:,j] / sds[j]
    return norm_mat
