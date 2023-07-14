# To the extent possible under law, the authors have waived all copyright and related or neighboring rights 
# to the CLEAN project governance document as per the CC-0 public domain dedication license.
# (TODO: governance document)
# 
# Written by Sanweda Mahagabin and Logan Grosenick, 2023.

# This is a command-line callable function to run the CLEAN preprocessing pipeline 
# based on a config file with the desired preprocessing parameters.

# Add citations for: MNE-Python, ICALabel, wICA, AutoReject, ASR, FASTER

# Import major libraries
import os, time, sys, re, glob
from datetime import datetime
import subprocess
import numpy as np
import h5py
import multiprocessing
import scipy.io as io
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import scipy
import pywt
import seaborn as sns

# Import MNE Python functions
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
from autoreject import AutoReject, Ransac

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
pipeline_log = logging.getLogger('CLEAN_preprocessing_pipeline')

# Load dotenv for credentials ?
# from dotenv import load_dotenv
# load_dotenv()

class Preprocessing(object):
    '''
    This class sequentially runs:
      (1) Data loading 
      (2) HP filtering 
      (3) Line noise removal 
      (3) Channel-wise cleaning (Wavelet ICA cleaning, ICLabel ICA cleaning, etc.)
      (5) Segmenting-wise rejection and interpolation
      (6) Re-referencing

    And generates plots and logs to evaluate each step. 

    It takes input arguments:
      parameters_file: a configuration file specifying parameters of the prediction run.

    '''
    def __init__(self, parameters_file):
       '''
       Initialize the pipeline. 
       '''
       self.date = datetime.now().date().isoformat()
       self.load_parameters(parameters_file)

       # Set bad channels container
       self.bad_channels_list=['VREF']

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

       # Set up the console handler, which logs INFO and higher level messages.                                                                                            
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
                     
    def load_parameters(self, parameters_file):
        ''' 
        Looks for parameter file in run directory.
        If it exists class parameters are set to match those of the file,
        otherwise an exception is returned.
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
            # path parameters
            self.input_path = parameters.get('paths', 'input_path')
            self.results_path = parameters.get('paths', 'results_path')

            # data type parameters
            self.data_from_pre = eval(parameters.get('data_from', 'pre'))
            self.data_from_post = eval(parameters.get('data_from', 'post'))
            self.data_from_resting_state = eval(parameters.get('data_from', 'resting_state'))
            self.data_from_TMS = eval(parameters.get('data_from', 'TMS'))
            self.data_from_motor = eval(parameters.get('data_from', 'motor'))

            # filtering parameters
            self.filter_raws_separately = eval(parameters.get('filtering', 'filter_raws_separately'))
            self.filter_notch = eval(parameters.get('filtering', 'notch'))
            self.notch_freqs = np.array(parameters.get('filtering', 'notch_freqs').strip('][').split(',')).astype(float)
            self.high_pass_cutoff = float(parameters.get('filtering', 'high_pass_cutoff'))
            self.filter_band_pass = eval(parameters.get('filtering', 'band_pass'))
            self.band_pass_low = float(parameters.get('filtering', 'band_pass_low'))
            self.band_pass_high = float(parameters.get('filtering', 'band_pass_high'))
            self.filter_type = str(parameters.get('filtering', 'filter_type'))

            # data cleaning parameters 
            self.wICA = eval(parameters.get('cleaning', 'wICA'))
            self.wICA_num_components = int(parameters.get('cleaning', 'wICA_num_components'))
            self.icalabel = eval(parameters.get('cleaning', 'icalabel'))
            self.iclabel_num_components = int(parameters.get('cleaning', 'iclabel_num_components'))
            self.bad_segment_interpolation = eval(parameters.get('cleaning', 'bad_segment_interpolation'))
            self.segment_interpolation_method = parameters.get('cleaning', 'segment_interpolation_method')
        except Exception as e:
            print('Failure loading parameters with error:',e)       
            pipeline_log.info("  An error occured when loding parameters: " +str(e))
            sys.exit(1)

        # Plot parameters -- UNDER CONSTRUCTION
        plt.style.use("ggplot")
        np.seterr(invalid='ignore')
        sns.set_style("white")
        sns.set_context("paper")
        sns.set_palette("deep")
        plt.rcParams['axes.linewidth'] = 0.5
        plt.rcParams['figure.dpi'] = 100

        # MNE output verbose-ness
        mne.set_log_level(parameters.get('logging', 'mne_log_level'))  

        self.parameters_file = parameters_file

    def load_data(self):
        '''
        Data loading fuction: TODO: add MNE loading and docstring.
        '''
        # Decide which data to load using values from parameter file
        if self.data_from_resting_state:
            if self.data_from_pre:
                filenames = [f for f in os.listdir(self.input_path) if 'reststate' and 'pre' in f]
            if self.data_from_post:
                filenames.extend([f for f in os.listdir(self.input_path) if 'reststate' and 'post' in f])
        elif self.data_from_TMS:
            filenames = [f for f in os.listdir(self.input_path) if 'treatment' in f]
        elif self.data_from_motor:
            filenames = [f for f in os.listdir(self.input_path) if 'motor' in f]
        else:
            raise ValueError('Data type is misspecified or does not exist in the input folder.')

        # Iterate over filenames in list
        for i,f in enumerate(filenames):
            data_raw_file = os.path.join(self.input_path, f)
            raw = mne.io.read_raw_egi(data_raw_file, preload=True, verbose='warning') # MNE 1.4.2
            raw.info["bads"] = self.bad_channels_list
            pipeline_log.info('')
            pipeline_log.info('  Loaded: '+f)

            # If we are filtering each raw dataset separately, iterated over each filtering and appending,
            # otherwise append the unfiltered data into one file and then notch and HP filter the full concatenated data.
            if self.filter_raws_separately and self.filter_notch:
                pipeline_log.info('  Notch at 60Hz.')
                pipeline_log.info('  High pass at '+str(self.high_pass_cutoff)+'Hz.')
                pipeline_log.info('')
                if i==0:
                    data = notch_and_hp(raw, l_freq=self.high_pass_cutoff, notch_freqs=self.notch_freqs, filter_type=self.filter_type)
                else:
                    data = mne.concatenate_raws([data,notch_and_hp(raw, l_freq=self.high_pass_cutoff, 
                            notch_freqs=self.notch_freqs, filter_type=self.filter_type)], verbose='warning')
            else:
                if i==0:
                    data = raw
                else:
                    data = mne.concatenate_raws([data,raw], verbose='warning')
        if not self.filter_raws_separately and self.filter_notch:
            pipeline_log.info('  Filtering concatenated data.')
            pipeline_log.info('  -->Notch filtered at 60Hz.')
            pipeline_log.info('  -->High pass filtered at'+str(self.high_pass_cutoff)+'Hz.')
            pipeline_log.info('')
            data = notch_and_hp(data)

        # Finally if band_pass is set to True, we bandpass the data at the values specified in the parameters file.
        if self.filter_band_pass:
            self.data = data.filter(l_freq=self.band_pass_low, h_freq=self.band_pass_high, method=self.filter_type, verbose='warning')

        # Save PSD of loaded and filtered data
        save_psd(pjoin(self.results_path+'prefiltered.png'), self.data)

        # TODO: Add other data stats to logger
        pipeline_log.info('  Data shape: '+str(self.data._data.shape))
        
    def run(self):
        # Clear screen and print basic data paths and parameter file location
        co.clear_screen()
        pipeline_log.info((co.color('purple', co.bold_text('CLEAN preprocessing pipeline.'))))
        pipeline_log.info('')
        pipeline_log.info(co.color('green','Paths'))
        pipeline_log.info((co.color('white','  Configuration file: ')) + self.parameters_file)
        pipeline_log.info((co.color('white','  Output Directory: ')) + self.results_path)
        pipeline_log.info('')

        ## Run through pipline steps, printing successful stages and settings to console and log
        # Load data files provided as list of paths to MFF files
        pipeline_log.info(co.color('periwinkle','Loading and filtering specified data from input directory...'))
        self.load_data()
        pipeline_log.info('')

        # Re-reference data
        pipeline_log.info(co.color('periwinkle','Re-referencing data to average reference...'))
        self._reference()
        pipeline_log.info('')

        # TODO: Add downsampling option here

        # Run ICALabel to detect bad ICs
        if self.icalabel:
            pipeline_log.info(co.color('periwinkle','Running ICALabel cleaning (this may take some time)...'))
            self._iclabel()
            pipeline_log.info((co.color('white','  Finished ICALabel.')))
        else: 
            pipeline_log.info(co.color('periwinkle','Skipping ICALabel...'))
        pipeline_log.info('')

        # Run wavelet ICA
        if self.wICA:
            pipeline_log.info(co.color('periwinkle','Running wavelet ICA cleaning with '+str(self.wICA_num_components)+' components (this may take some time)...'))
            self._wICA()
            pipeline_log.info((co.color('white','  Finished wICA.')))
        else:
            pipeline_log.info(co.color('periwinkle','  Skipping wICA...'))
        pipeline_log.info('')

        # Segment data into epochs, identify remaining bad segments, and interpolate
        if self.bad_segment_interpolation:
            pipeline_log.info(co.color('periwinkle','Identifying and interpolating bad segments...'))
            self._interpolate_bad_segments()
            pipeline_log.info((co.color('white','  Finished bad segment identification and interpolation.')))
        else:
            pipeline_log.info((co.color('white','  Skipping bad segment identification and interpolation.')))
        pipeline_log.info('')

    def _notch_and_hp(self):
        self.data = notch_and_hp(self.data)

        # Save the filtered data to raw MNE-Python file 
        self.data.save(pjoin(self.results_path,'filtered.fif'), overwrite=True)

        # Plot filtered data artifacts
        save_eog_plot(pjoin(self.results_path,'eog_filtered.png'), self.data)

    def _wICA(self):
        ica = FastICA(n_components=self.wICA_num_components, whiten="arbitrary-variance")
        ica.fit(self.data.get_data().T)  
        ICs = ica.fit_transform(self.data.get_data().T)

        self.wICs, self.artifacts = wICA(ica, ICs)
        self.data._data -= self.artifacts.T

        # Save sparkline plots of the IC timeseries pre_ICA, and the resulting thresholded IC timeseries.
        save_sparkline_ica(pjoin(self.results_path,'ICA_timeseries_pre_wICA.png'), ICs) 
        save_sparkline_ica(pjoin(self.results_path,'wICA_artifact_timeseries.png'), self.wICs) 

        # Save the wavelet-ICA-cleaned raw MNE-Python file 
        self.data.save(pjoin(self.results_path,'wICA_cleaned.fif'), overwrite=True)

    def _iclabel(self):
        # Run ICALabel on the current data
        self.ic_labels, self.ic_ica, self.ic_cleaned = iclabel(self.data, num_components=self.iclabel_num_components)

        # Save ICA topoplots and a folder of individual IC statistic plots
        save_ica_components(pjoin(self.results_path,'ICALabel_ICA_topoplots.png'), self.ic_ica, ic_label_list=self.ic_labels['labels'])
        ic_save_path = pjoin(self.results_path,'ICALabel_ICA_topoplots/')
        if not os.path.exists(ic_save_path):
           os.makedirs(ic_save_path)
        for i,label in enumerate(self.ic_labels['labels']):
            save_single_ic_plot(pjoin(ic_save_path,'IC'+str(i).zfill(3)+'.png'), self.ic_ica, self.data, component_number=i, ic_label=self.ic_labels['labels'][i])
            plt.close()

        # Generate artifact plots for cleaned data and data without cleaning
        save_eog_plot(pjoin(self.results_path,'eog_icalabel_precleaning.png'), self.data)
        save_eog_plot(pjoin(self.results_path,'eog_icalabel_cleaned.png'), self.ic_cleaned)

        # Replace pipeline data with new cleaned data
        self.data = self.ic_cleaned

        # Save the ICALabel-cleaned raw MNE-Python file 
        self.ic_cleaned.save(pjoin(self.results_path,'icalabel_cleaned.fif'), overwrite=True)

    def _reference(self, reference_method='average'):
        self.data = self.data.set_eeg_reference(reference_method)

    def _interpolate_bad_segments(self):
        self.data = interpolate_bad_segments(self.data, method=self.segment_interpolation_method)
        # Add bad segment plots

#------------- Filtering functions -------------#
def notch_and_hp(raw, notch_freqs, l_freq=1.0, h_freq=None, filter_type='fir'):
    notch_freqs = np.array(notch_freqs)
    raw_notch = raw.copy().notch_filter(freqs=notch_freqs, verbose='warning')
    raw_hp = raw_notch.filter(l_freq=l_freq, h_freq=h_freq, method=filter_type, verbose='warning')
    return raw_hp

#------------- Wavelet ICA functions -------------#
def ddencmp(x, wavelet='coif5', scale=1.0):
    '''Python recreation of MATLABs ddencmp function for choosing wavelet thresholds'''
    n = len(x)
    (cA, cD)  = pywt.dwt(x,wavelet)
    noiselev = np.median(np.abs(cD))/0.6745
    thresh = np.sqrt(2*np.log(len(x)))*noiselev*scale
    return thresh

def wICA(ica, ICs, levels=7, wavelet='coif5', normalize=False, 
         trim_approx=False, thresholding='soft', verbose=True):
    '''
    Wavelet ICA thresholding. 
    Args:
        ica: fitted ica object (currently only works with Sklearn's FastICA). TODO: add MNE options.
        ICs: a numpy array of independent components with shape: (#time points, #components).
        levels: number of wavelet levels.
        wavelet: wavelet to use, defaults to Coiflet 5.
        normalize: boolean, see pywt.swt documentation
        trim_approx: boolean, see pywt.swt documentation
        thresholding: 'hard' or 'soft',  should wavelet thresholding use a hard or soft threshold.
        verbose: boolean flag for verbose printing.
    Returns:
        wICs: the wavelet thresholded ICs, leaving only large amplitude effects.
        artifacts: the wICs ICA inverse transformed back to ambient data space, shape: (#time points, #channels). 
    '''
    # Pad out the data to the correct length for wavelet thresholding
    modulus = np.mod(ICs.shape[0],2**levels)
    if modulus !=0:
        extra = np.zeros((2**levels)-modulus)
    if verbose:
        print('Data padded with ', str(len(extra)), ' additional values.')
        print('Wavelet thresholding with wavelet:', wavelet)
    wavelet = pywt.Wavelet(wavelet)

    # Iterate over independent components, thresholding each one using a stationary wavelet transform
    # with soft thresholding. 
    print('  Fitting ICA for wavelet-ICA cleaning...')
    wICs = []
    for i in range(ICs.shape[1]):
        if verbose:
            print("Thresholding IC#",i)
        sig = np.concatenate((ICs[:,i],extra))
        thresh = ddencmp(sig, wavelet)
        swc = pywt.swt(sig, wavelet, level=levels, start_level=0, trim_approx=trim_approx, norm=normalize)
        y = pywt.threshold(swc, thresh, mode=thresholding, substitute=0)
        wICs.append(pywt.iswt(y, wavelet, norm=normalize))
    wICs = np.asarray(wICs)[:,0:ICs.shape[0]].T
    artifacts = ica.inverse_transform(ICs)
    return wICs, artifacts

#------------- ICA Label functions -------------#
def iclabel(mne_raw, num_components=20, keep=["brain", "other"], method='infomax', fit_params=dict(extended=True)):
    '''
    Uses the MNE-ICALabel method to classify independent components into six different estimated sources of 
    variability: brain, muscle artifact, eye blink, heart beat, line noise, channel noise, and other.
    Args:
        mne_raw: a data file in the MNE-Python raw format.
        n_components: the number of ICA components to estimate. 
        method: the ICA method to use, see MNE documentation for more information.
        fit_params: additional arguments passed to the ICA method. 
        keep: a list of source types to keep; 
              options are 'brain', 'muscle artifact', 'eye blink', 
              'heart beat', 'line noise', 'channel noise', and 'other'.
    Returns:
        ic_labels: a set of labels for the ICs estimated.
        ica: a fit MNE ICA object.
        cleaned: a new data file with the all but the data categories in 'keep' removed.
    '''
    # Fit ICA to mne_raw
    print('  Fitting ICA with '+str(num_components)+' components.')
    ica = ICA(n_components=num_components, max_iter="auto", method=method, random_state=42, fit_params=fit_params)
    ica.fit(mne_raw)
    
    # Use label_components function from ICLabel library to classify ICs
    print('  Using ICALabel to classify independent components.')
    ic_labels = label_components(mne_raw, ica, method="iclabel")
    labels = ic_labels["labels"]
    
    # Exclude ICs not in 'keep' list and reconstruct cleaned data
    exclude_idx = [idx for idx, label in enumerate(labels) if label not in keep]
    print(f"  Excluding these ICA components: {exclude_idx}")
    reconst_raw = mne_raw.copy()
    cleaned = ica.apply(reconst_raw, exclude=exclude_idx)
    return ic_labels, ica, cleaned

def interpolate_bad_segments(raw, segment_length=1.0, method='autoreject'):
    # generate epochs object from raw
    raise NotImplemented()

    # Use mne-compatible autoreject library to clean epoched data
    if method=='autoreject':
        ar = AutoReject()
    elif method=='ransac':
        ar = Ransac()
    else:
        raise pipeline_log.ValueError("Specified bad segment method not implemented. Current options are 'autoreject' and 'ransac'.")
    epochs_clean = ar.fit_transform(epochs)  

def artifact_subspace_reconstruction(raw):
    '''
    Apply artifact subspace reconstrution method using a Python implementation of EEGLAB's clear_rawdata ASR method. 
    See: https://github.com/DiGyt/asrpy.  
    '''
    import asrpy
    asr = asrpy.ASR(sfreq=raw.info["sfreq"], cutoff=20)
    asr.fit(raw)   
    raw = asr.transform(raw)
    return raw

#------------- Plotting functions -------------#
def save_psd(save_file, mne_raw, xlim=[0,100]):
    '''
    Save a power spectral density plot of the time series in mne_raw to the specified absolute file path
    using MNE-Python's compute_psd function.

    '''
    fig = mne_raw.compute_psd().plot(show=False)
    axs = fig.get_axes()
    axs[0].set_xlim(xlim)
    sns.despine()
    plt.savefig(save_file)

def eeg_sparkline(timeseries,x=None,colors=None,normalize=True,color='black', alpha=0.5,
                     lwd=1.0,scale_factor=1.0, spacing = 1.3, figsize=(15,10), ax=None):
    '''                                                                                                                  
    Plot a simple sparkline plot of EEG time series in the rows of the provided matrix. 
    Input Args:
        timeseries: numpy array with shape (n_channels, n_timepoints).
        x: optional labels for the x-axis, must be of same length as n_timepoints.
        colors: optional vector of colors for sparkline traces.
        normalize: boolean, if true use mean or max normalization.
        color: default color for all sparkline traces.
        alpha: alpha value for all sparkline traces.
        lwd: linewidth for all sparkline traces.
        scale_factor: scaling factor for sparkline trace amplitudes (if normalized).
        spacing: allows control over space between sparkline traces.
        figsize: passed to matplotlib to control figure size.
        ax: optional matplotlib axes object for overplotting.
    Returns:
        ax: matplotlib axes object.
                                                                                                                         
    '''
    if ax is None:
        f = plt.figure(figsize=figsize)
        ax = f.add_subplot(111)
    timeseries = np.flipud(timeseries)
    if x is None:
        x = np.array(range(timeseries.shape[1]))
    for c in range(timeseries.shape[0]):
        ts = timeseries[c, :]
        if normalize == True or normalize == 'mean':
            ts = (ts - ts.mean()) / np.abs(ts.mean())*scale_factor
        if normalize == 'max':
            ts = (ts - ts.mean()) / np.abs(ts).max()*scale_factor
        if colors is not None:
            ax.plot(x, ts + c*spacing, linewidth = lwd, color = colors[c], alpha=alpha)
        else:
            ax.plot(x, ts + c*spacing, linewidth = lwd, color = color, alpha=alpha) #'#0000ff')                 
        plt.xlim([0,np.max(x)])
        ax.spines[['left','right', 'top']].set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    return ax

def save_sparkline_ica(save_file, ica_timeseries):
    '''
    Save a sparkline plots of independent component timeseries (n_timepoints, n_ICs) to the specified absolute file path
    using function 'eeg_sparkline'.

    '''
    num_components = ica_timeseries.shape[1]
    ax = eeg_sparkline(ica_timeseries.T, scale_factor=1, normalize='max', alpha=0.7, figsize=(25, num_components))
    plt.xlabel('Time (samples)')
    plt.savefig(save_file)

def save_ica_components(save_file, ica, ic_label_list=None):
    '''
    Save a set of independent component plots using MNE-Python's plot_components method for ICA objects.
    Args:
        save_file: absolute path to save PNG output file.
        ica: an MNE-Python ica object.
        ic_label_list: optional list of conformal labels from ica_label to add to plot topomap titles.
    
    '''
    fig = ica.plot_components(show=False)
    if ic_label_list is not None:
        axs = fig.get_axes()
        assert len(axs)==len(ic_label_list)
        for i,ax in enumerate(axs):
            current_title = ax.get_title()
            ax.set_title(current_title+':\n '+ic_label_list[i])
    plt.savefig(save_file)

def save_single_ic_plot(save_file, ica, mne_raw, component_number=0, ic_label=None):
    '''
    Save a summary independent component plot using MNE-Python's plot_properties method for ICA objects.
    Args:
        save_file: absolute path to save PNG output file.
        ica: an MNE-Python ica object.
        mne_raw: an MNE-Python raw object.
        component_number: the independent component index to plot. 
        ic_label: optional label from ica_label to add to plot topomap title.
    
    '''
    fig = ica.plot_properties(mne_raw, picks=[component_number], verbose=False, show=False)
    if ic_label is not None: 
        axs = fig[0].get_axes()
        current_title = axs[0].get_title()
        axs[0].set_title(current_title+': '+ic_label)
    plt.savefig(save_file)

def save_eog_plot(save_file, mne_raw, channel_list=['E32','E241','E25','E238']):
    '''
    Save an eog artifact plot for the time series in mne_raw to the specified absolute file path.
    Uses MNE-Python's create_eog_epochs and plot_joint functions and defaults to channels 
    ['E32','E241','E25','E238'] which correspond to EGI's 256 channel hydrocel net eye motion channels.

    '''
    average_ecg = mne.preprocessing.create_eog_epochs(mne_raw,ch_name=channel_list).average()
    average_ecg.plot_joint(show=False)
    sns.despine()
    plt.savefig(save_file)
        
#-------------------------------------------------------------------------------                                     
# Main                                                                                                               

if __name__ == '__main__':
    # Parse command line                                                                                             
    from argparse import ArgumentParser
    from argparse import SUPPRESS
    parser = ArgumentParser()

    pred_pipeline = Preprocessing('test_config.cfg')
    pred_pipeline.run()
