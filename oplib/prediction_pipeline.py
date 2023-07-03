# Copyright (C) Responsive Capital Management - All Rights Reserved.
# Unauthorized copying of any code in this directory, via any medium is strictly prohibited.
# Proprietary and confidential.
# Written by Logan Grosenick <logan@responsive.ai>, 2019.

# Command-line callable function to run full analysis pipeline based on config file with analysis parameters.

# Import major libraries
import os, time, sys, re, glob
from datetime import datetime
import subprocess
import numpy as np
import h5py
import multiprocessing
import scipy.io as io

# Import local functions
from util.misc import run_command_wrapper
from util.email_sender import send_email
import util.console_output as co

# Alias useful functions
pjoin = os.path.join

# Set up the pipeline log.  It is configured in the constructor below,
# but used by the free functions outside of the class, hence its        
# definition in global scope here.      
import logging
pipeline_log = logging.getLogger('prediction_pipeline')

from dotenv import load_dotenv
load_dotenv()

class PredictionPipeline(object):
    '''
    This class sequentially runs:
      (1) Data loading and indexing
      (2) Feature engineering
      (3) Prediction

    It takes input arguments:
      parameters_file: a configuration file specifying parameters of the prediction run.

    '''
    def __init__(self, parameters_file):
       '''
       Initialize the pipeline. 
       '''
       self.date = datetime.now().date().isoformat()
       self.load_parameters(parameters_file)

       # Set up a logfile for this run.  We append the date to the 
       # logfile, so each run of the pipeline is unique.
       pipeline_log.setLevel(logging.DEBUG)
       from time import localtime, strftime
       self.log_path = pjoin(self.results_path, 'logs')
       if not os.path.exists(self.log_path): os.makedirs(self.log_path)
       self.local_logfile = pjoin(self.log_path, strftime("%Y-%b-%d-%H:%M:%S_log.txt", localtime()))

       fh = logging.FileHandler(self.local_logfile)
       fh.setLevel(logging.DEBUG)
       format = '%(asctime)s - %(levelname)s - %(message)s'
       fh_formatter = logging.Formatter(format)
       fh.setFormatter(fh_formatter)
       pipeline_log.addHandler(fh)

       # Set up the console handler, which logs INFO and higher                                                     
       # level messages.                                                                                            
       ch = logging.StreamHandler()
       ch.setLevel(logging.INFO)
       ch_formatter = logging.Formatter('%(message)s')
       ch.setFormatter(ch_formatter)
       pipeline_log.addHandler(ch)
       
       # Make a copy of the parameters file used for this run
       import shutil
       param_path = pjoin(self.results_path, 'parameters')
       paramcopy_file = pjoin(param_path, strftime("%Y-%b-%d-%H:%M:%S-params.cfg", localtime()))

       if not os.path.exists(param_path): os.makedirs(param_path)
       shutil.copyfile(parameters_file, paramcopy_file)

       # Initialize other default parameters
       self.select_idx = None
       self.y_thresh = 0.0
       self.window_size = 10
       
    def load_parameters(self, parameters_file):
        ''' 
        Looks for parameter file in run directory.
        If it exists class parameters are set to match those of the file,
        otherwise an exception is returned
        '''
        # Import and instantiate ConfigParser.   
        import configparser
        parameters = configparser.ConfigParser()
        parameters.read(parameters_file)
        if len(parameters.sections()) == 0:
            print(len(parameters.sections()))
            print('Could not load parameters from ', parameters_file)
            sys.exit(1)
        try:
            # paths
            self.results_path = parameters.get('paths', 'results_path')
            # dataio
            self.train_range = eval(parameters.get('dataio','train_range'))
            self.test_range = eval(parameters.get('dataio','test_range'))
            self.user_ids = eval(parameters.get('dataio', 'user_ids'))
            self.feature_ids = eval(parameters.get('dataio','feature_ids'))
            self.num_clients = eval(parameters.get('dataio', 'num_clients'))
            self.randomize_clients = parameters.get('dataio', 'randomize')
            
        except Exception as e:
            print('Failure loading parameters with error:',e)       
            pipeline_log.info("  An error occured when loding parameters: " +str(e))
            sys.exit(1)
        self.parameters_file = parameters_file

    def load_data(self):
        import dataio

        LOCAL_POSTGRES_HOST = os.getenv('PG_HOST') or 'op1d70bl9z8du9y.c9x7afofaq1k.eu-central-1.rds.amazonaws.com'
        db_cursor = dataio.get_cursor(LOCAL_POSTGRES_HOST)

        # Load data
        pipeline_log.info('Loading data for model training...')
        taxonomy_data = dataio.load_taxonomy_data(db_cursor,clients=self.num_clients,time_range=self.train_range)
        # Get feature look up table 
        self.feature_LUT = dataio.get_feature_LUT(db_cursor)
        # Load data into data object
        self.data = dataio.Data(taxonomy_data,self.feature_LUT)

        # XXX TODO turn these and all other console prints to pipeline_log messages
        print("\t "+str(len(self.data.features_unique))+" unique features on "+
              str(len(self.data.users_unique))+" users over "+str(len(self.data.times_unique))+
              " time points ("+str(self.data.times_unique[0]),'to '+str(self.data.times_unique[-1])+').')

        # Load out of sample data
        pipeline_log.info('Loading out-of-sample data for model prediction...')
        taxonomy_data_oos = dataio.load_taxonomy_data(db_cursor,clients=self.data.users_unique.tolist(),
                                                      features=self.data.features_unique.tolist(),
                                                      time_range=self.test_range,
                                                      time_window=self.window_size)
        # Load OOS data into data object
        self.data_oos = dataio.Data(taxonomy_data_oos,self.feature_LUT,features_unique=self.data.features_unique)
        print("\t "+str(len(self.data_oos.features_unique))+" unique features on "+
              str(len(self.data_oos.users_unique))+" users over "+str(len(self.data_oos.times_unique))+
              " time points ("+str(self.data_oos.times_unique[0]),'to '+
              str(self.data_oos.times_unique[-1])+').')
        
    def get_features(self,feature_num=270178178,k_ahead_stat='sum'):
        import features
        # Get windowed features and prediction targets
        print('Data features:')
        X, Y = features.get_windowed_features(self.data,n_users=self.data.tensor.shape[0],
                                              window_len=self.window_size,
                                              k_ahead=4,xcorr=False,stats=True,savepath=None)
        # Get windowed features and prediction targets OOS
        print('Out-of-sample (OOS) data features:')
        Xoos, Yoos = features.get_windowed_features(self.data_oos,n_users=self.data_oos.tensor.shape[0],
                                              window_len=self.window_size,
                                              k_ahead=4,xcorr=False,stats=True,
                                                    savepath=None)
        # Feature quality control check
        # XXX TODO: run for data and oos data (pass metrics from data run to oos data run)
        print('Feature quality control metrics:')
        features.feature_QC(self.data,sigma=3.5,window_size=10)

        # X: (clients, time, win stats, categories )
        # Y: (clients, time, win_data, categories ))
        if k_ahead_stat == 'sum':
            Y = np.sum(np.asarray(Y),axis=2)
            Yoos = np.sum(np.asarray(Yoos),axis=2)
            
        print("Prediction target:", self.feature_LUT[feature_num])
        f_idx = np.where(self.data.features_unique==feature_num)[0]
        y = np.squeeze(Y[:,:,f_idx])
        yoos = np.squeeze(Yoos[:,:,f_idx])
        
        # TODO: write test for reshape
        self.X = np.asarray(X)
        self.Xoos = np.asarray(Xoos)
        self.X_2d = self.X.reshape((self.X.shape[0]*self.X.shape[1],self.X.shape[2]*self.X.shape[3]))
        self.Xoos_2d = self.Xoos.reshape((self.Xoos.shape[0]*self.Xoos.shape[1],self.Xoos.shape[2]*self.Xoos.shape[3]))
        
        self.y = np.abs(np.asarray(y)) # note sign shift
        self.yoos = np.abs(np.asarray(yoos)) # note sign shift
        self.y_2d = self.y.reshape((self.y.shape[0]*self.y.shape[1]))
        self.y_2dbin = features.binarize(self.y_2d, self.y_thresh)
        self.yoos_2d = self.yoos.reshape((self.yoos.shape[0]*self.yoos.shape[1]))
        self.yoos_2dbin = features.binarize(self.yoos_2d, self.y_thresh)

    def setup_validation(self,num_folds=10,train_fraction=0.6,validation_fraction=0.0):
        # Get CV folds
        from util import validation
        self.CV_folds = validation.get_CV_folds(self.X_2d.shape[0],self.y_2dbin, folds=num_folds,
                                                train_fraction=train_fraction,
                                                validation_fraction=validation_fraction)
        # Print validation settings to screen
        print('Running',str(num_folds)+'-fold cross validation:')
        print('\t Training fraction:', str(train_fraction*100)+'% of data ('+
              str(int(train_fraction*self.X_2d.shape[0])),'/',str(self.X_2d.shape[0]),' measurements.)')
        if validation_fraction > 0.0:
            print('\t Validation fraction:', str(validation_fraction*100)+'% of data ('+
                  str(int(validation_fraction*self.X_2d.shape[0])),'/',str(self.X_2d.shape[0]),' measurements.)')
        print('\t Test fraction:', str((1.0-train_fraction-validation_fraction)*100)+'% of data ('+
              str(int((1.0-train_fraction-validation_fraction)*self.X_2d.shape[0])),'/ '+
              str(self.X_2d.shape[0]),' measurements.)')

    def screen_features(self,input_shape=2,num_select=100):
        # If variables have been flattened to 2D # time-subject x variable-window/stats
        if input_shape == 2:
            import screening
            # Screen for event detection
            print('Binary screening (for event detection)...')
            self.select_ranks_event, self.select_idx_event = screening.simple_screen(self.X_2d, self.y_2dbin,
                                                                self.CV_folds, num_select=num_select,
                                                                                     selection=np.median)
            print('Continuous screening (magnitude prediction)...')
            print('\t y threshold:',str(self.y_thresh))
            self.select_ranks_mag, self.select_idx_mag = screening.simple_screen(self.X_2d, self.y_2d,
                                                                self.CV_folds, num_select=num_select,
                                                                                 selection=np.median)
        else:
            raise NotImplementedError('Screening not yet ipmlemented for input_shape != 2.')
        
    def embed_features(self):
        print('Embedding options not yet implemented')
        
    def predict_events(self, class_weight='balanced'):
        import event_detection

        # Use classification to detect nonzero events 
        from sklearn import svm, linear_model
        print('Fitting event classifier...')
        # Specify model
        clf = linear_model.SGDClassifier(loss='modified_huber',max_iter=1000, tol=1e-3, alpha=0.015,
                                         n_jobs=-1, class_weight=class_weight)#,early_stopping=True) 

        # Detect nonzero event times 
        self.event_results = event_detection.event_classifier(clf, self.X_2d, self.y_2dbin.ravel(), self.CV_folds,
                                                   select_idx=None,
                                                   rescale=True, sample_weights=None)
        # Print event detection results to screen
        event_detection.print_classification_results(self.event_results)

    def predict_magnitudes(self):
        '''
        Use regression to predict magnitude of events predicted by self.predict_events().
        '''
        import regression

        # Use Regression to estimate event magnitude of nenzero events
        print('Fitting event magnitide regression...')
        # Specify regression model
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import SGDRegressor
        #reg_model = SGDRegressor(loss='huber',max_iter=1000, tol=1e-3, alpha=1.0)
        reg_model = RandomForestRegressor(criterion='mae', max_depth=100, min_samples_split = 5, 
                                  min_samples_leaf = 1, max_features = 'sqrt', random_state=0, 
                                  n_estimators=500, n_jobs=-1)           

        # Predict event magnitudes
        self.magnitude_results = regression.event_regression(reg_model, self.X_2d, self.y_2d, self.CV_folds,
                                                        select_idx=self.select_idx_mag,
                                                        rescale=False, sample_weights=None)

        # Print event magnitude prediction results
        regression.print_magnitude_prediction(self.magnitude_results)
        regression.plot_magnitude_prediction(self.magnitude_results)
        
    def predict_heldout(self):
        pass

    def plot_results(self):
        pass

    def run(self):
        # Clear screen and print basic data paths and parameter file location
        co.clear_screen()
        pipeline_log.info((co.color('white',
                                    co.bold_text('Responsive Prediction '
                                                 'Pipeline.'))))
        pipeline_log.info('')
        pipeline_log.info(co.color('periwinkle','Paths'))
        pipeline_log.info((co.color('white','  Configuration file: ')) + self.parameters_file)
        pipeline_log.info((co.color('white','  Output Directory: ')) + self.results_path)
        pipeline_log.info('')

        ## Run through pipline steps, printing successful stages and settings to console and log
        # Load data
        pipeline_log.info(co.color('periwinkle','Loading data from database...'))
        self.load_data()
        pipeline_log.info('')

        # Generate features
        pipeline_log.info(co.color('periwinkle','Calculating engineered features...'))
        self.get_features()
        pipeline_log.info('')

        # Screen features for importance to reduce data dimensionality
        pipeline_log.info(co.color('periwinkle','Setting up validation and prediction...'))
        self.setup_validation()
        pipeline_log.info('')

        # Screen features for importance to reduce data dimensionality
        pipeline_log.info(co.color('periwinkle','Screening engineered features...'))
        self.screen_features()
        pipeline_log.info('')

        # Embed features in lower dimensional, potentially nonlineal space
        pipeline_log.info(co.color('periwinkle','Embedding engineered features...'))
        self.embed_features()
        pipeline_log.info('')

        # Detect and predict when events will be (many are zero)
        pipeline_log.info(co.color('periwinkle','Predicting event times... '))
        self.predict_events()
        pipeline_log.info('')

        # For predicted events, estimate their magnitude
        pipeline_log.info(co.color('periwinkle','Predicting event magnitudes... '))
        self.predict_magnitudes()
        pipeline_log.info('')

        # Generate summary statistics and plots for model assessment 
        pipeline_log.info(co.color('periwinkle','Generating summary plots... '))
        self.plot_results()
        pipeline_log.info('')

        
#-------------------------------------------------------------------------------                                     
# Main                                                                                                               

if __name__ == '__main__':
    # Parse command line                                                                                             
    from argparse import ArgumentParser
    from argparse import SUPPRESS
    parser = ArgumentParser()

    pred_pipeline = PredictionPipeline('test_config.cfg')
    pred_pipeline.run()
