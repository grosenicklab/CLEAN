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
import shutil
from datetime import datetime
import subprocess
import configparser
import gc
import numpy as np
import h5py
import multiprocessing
import scipy.io as io
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import scipy
import pywt
import seaborn as sns
from pyprep.find_noisy_channels import NoisyChannels

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
      (4) Bad segment rejection
      (5) Bad channel rejection
      (6) Channel-wise cleaning (Wavelet ICA cleaning, ICLabel ICA cleaning, etc.)
      (7) Re-referencing

    And generates plots and logs to evaluate each step. 

    It takes input arguments:
      parameters_file: a configuration file specifying parameters of the prediction run.

    '''
    def __init__(self, parameters_file):
        '''
        Initialize the pipeline using a parameters file. See test_config.cfg in this directory for an example.

        '''
        # Set date
        self.date = datetime.now().date().isoformat()

        # Set bad channels list
        self.bad_channels_list = ['VREF']

        # Load parameters from parameters_file
        self.load_parameters(parameters_file)

        # Make results path if it does not exist
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        # Make a results folder for this run
        tmp = os.listdir(self.results_path)
        if 'results_run' in ''.join(tmp):
            runs = [i for i in tmp if 'results_run' in i]
            runs.sort()
            next_run_num = int(runs[-1].split('_')[-1]) + 1
        else:
            next_run_num = 0
        results_foldername = 'results_run_'+str(next_run_num).zfill(3)
        self.results_savepath = pjoin(self.results_path, results_foldername)
        os.makedirs(self.results_savepath)

        # Set up a logfile for this run.  We append the date to the
        # logfile, so each run of the pipeline is unique.
        pipeline_log.setLevel(logging.DEBUG)
        self.log_path = pjoin(self.results_savepath, 'logs')
        if not os.path.exists(self.log_path): os.makedirs(self.log_path)
        self.local_logfile = pjoin(self.log_path, time.strftime("%Y-%b-%d-%H:%M:%S_log.txt", time.localtime()))

        # Ensure messages aren't passed to the root logger to prevent duplicate messages
        pipeline_log.propagate = False  

        # A flag to check if logging handlers have been set
        if not pipeline_log.handlers:
            # Set up a logfile for this run. We append the date to the logfile so each run is unique.
            pipeline_log.setLevel(logging.DEBUG)
            self.log_path = pjoin(self.results_savepath, 'logs')
            if not os.path.exists(self.log_path): os.makedirs(self.log_path)
            self.local_logfile = pjoin(self.log_path, time.strftime("%Y-%b-%d-%H:%M:%S_log.txt", time.localtime()))

            # Set up the logging handlers only once
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
        param_path = pjoin(self.results_savepath, 'parameters')
        paramcopy_file = pjoin(param_path, time.strftime("%Y-%b-%d-%H:%M:%S-params.cfg", time.localtime()))
        if not os.path.exists(param_path): os.makedirs(param_path)
        shutil.copyfile(parameters_file, paramcopy_file)

    def load_parameters(self, parameters_file):
        '''
        Looks for parameter file in run directory.
        If it exists class parameters are set to match those of the file,
        otherwise an exception is returned.
        '''
        # Import and instantiate ConfigParser.
        parameters = configparser.ConfigParser()
        parameters.read(parameters_file) 
        if len(parameters.sections()) == 0:
            print(len(parameters.sections()))
            print('Could not load parameters from ', parameters_file)
            sys.exit(1)
        try:
            # Map string log levels to logging module constants
            log_level_map = {
                'DEBUG': logging.DEBUG,
                'INFO': logging.INFO,
                'WARNING': logging.WARNING,
                'ERROR': logging.ERROR,
                'CRITICAL': logging.CRITICAL
            }

            # general setting parameters
            self.mne_log_level = parameters.get('general', 'mne_log_level').upper()

            # Set the log level if it's a recognized level, otherwise raise an error
            if self.mne_log_level in log_level_map:
                pipeline_log.setLevel(log_level_map[self.mne_log_level])
            else:
                raise ValueError(f"Unknown log level: {self.mne_log_level}")

            self.memory_intensive = eval(parameters.get('general', 'memory_intensive'))

            # path parameters
            self.input_path = parameters.get('paths', 'input_path')
            self.results_path = parameters.get('paths', 'results_path')

            # data type parameters
            self.data_types = {
                'pre-treatment': eval(parameters.get('data_from', 'pre')),
                'post-treatment': eval(parameters.get('data_from', 'post')),
                'resting_state': eval(parameters.get('data_from', 'resting_state')),
                'TMS': eval(parameters.get('data_from', 'TMS')),
                'motor': eval(parameters.get('data_from', 'motor')),
                'fif': eval(parameters.get('data_from', 'fif')),
            }

            # filtering parameters
            self.filter_raws_separately = eval(parameters.get('filtering', 'filter_raws_separately'))
            self.filter_notch = eval(parameters.get('filtering', 'notch'))
            self.notch_freqs = np.array(parameters.get('filtering', 'notch_freqs').strip('][').split(',')).astype(float)
            self.notch_widths = parameters.get('filtering', 'notch_widths').strip('][').split(',')
            for i, nw in enumerate(self.notch_widths):
                try:
                    if nw.strip() == 'None':
                        self.notch_widths[i] = None
                    else:
                        self.notch_widths[i] = float(nw)
                except Exception as e:
                    print(f"Error converting notch_width {nw}: {e}")

            self.notch_withs = np.array(self.notch_widths)
            self.filter_band_pass = eval(parameters.get('filtering', 'band_pass'))
            self.band_pass_low = float(parameters.get('filtering', 'band_pass_low'))
            self.band_pass_high = float(parameters.get('filtering', 'band_pass_high'))
            self.filter_type = str(parameters.get('filtering', 'filter_type'))

            # resampling parameters
            self.resample = eval(parameters.get('resampling', 'resample'))
            self.resample_rate = float(parameters.get('resampling', 'resampling_rate'))

            # data cleaning parameters
            self.known_bad_channels = eval(parameters.get('cleaning', 'known_bad_channels'))
            self.bad_channels_list.extend(self.known_bad_channels)
            self.screen_bad_channels = eval(parameters.get('cleaning', 'screen_bad_channels'))
            self.wICA = eval(parameters.get('cleaning', 'wICA'))
            self.wICA_num_components = int(parameters.get('cleaning', 'wICA_num_components'))
            self.icalabel = eval(parameters.get('cleaning', 'icalabel'))
            self.iclabel_num_components = int(parameters.get('cleaning', 'iclabel_num_components'))
            self.asr = eval(parameters.get('cleaning', 'asr'))
            self.asr_cutoff = float(parameters.get('cleaning', 'asr_cutoff'))
            self.bad_segment_interpolation = eval(parameters.get('cleaning', 'bad_segment_interpolation'))
            self.segment_interpolation_method = parameters.get('cleaning', 'segment_interpolation_method')

        except Exception as e:
            print('Failure loading parameters with error:', e)
            pipeline_log.info("  An error occured when loading parameters: " + str(e))
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
        mne.set_log_level(parameters.get('general', 'mne_log_level'))

        self.parameters_file = parameters_file

    def load_and_filter_data(self):
        '''
        Data loading fuction: TODO: add MNE loading and docstring.
        '''

        def load_file(filename):
            data_raw_file = os.path.join(self.input_path, filename)
            if not self.data_types['fif']:
                raw = mne.io.read_raw_egi(data_raw_file, preload=True, verbose='warning')  # MNE 1.4.2
            else:
                raw = mne.io.read_raw_fif(data_raw_file, preload=True)

            # Set bad channels list to include those specified in class loading (e.g. VREF)
            raw.info["bads"] = self.bad_channels_list
            pipeline_log.info(f'Loaded: {filename}')

            # Save PSD of loaded and unfiltered data
            save_psd(pjoin(self.results_savepath, 'psd_unfiltered' + filename.split('_')[3] + '_' + filename.split('_')[4] + '.png'), raw)

            return raw

        def filter_and_concatenate(filenames, data_type):
            # Sort filenames to ensure file concatenation occurs in order of data collection
            if data_type == 'pre-treatment' or data_type == 'post-treatment':
                try:
                    filenames = sorted(filenames, key=lambda f: int(re.search(r'reststate(\d+)', f).group(1)))
                except AttributeError:
                    raise ValueError("Filename does not contain 'reststateX' format")

            def apply_filtering(raw_data):
                # Applies filtering to raw data based on class parameters.
                if self.filter_notch and self.filter_band_pass:
                    pipeline_log.info(f'--> {self.notch_freqs[0]} Hz notch filter applied.')
                    pipeline_log.info(f'--> {self.band_pass_low} - {self.band_pass_high} Hz band pass filter applied.')
                    return notch_and_hp(raw_data, l_freq=self.band_pass_low, h_freq=self.band_pass_high,
                                        notch_freqs=self.notch_freqs, notch_widths=self.notch_widths,
                                        filter_type=self.filter_type)
                elif self.filter_band_pass:
                    pipeline_log.info(f'--> {self.band_pass_low} - {self.band_pass_high} Hz band pass filter applied.')
                    return notch_and_hp(raw_data, l_freq=self.band_pass_low, h_freq=self.band_pass_high, 
                                        notch_freqs=None, notch_widths=None, filter_type=self.filter_type)
                elif self.filter_notch:
                    pipeline_log.info(f'--> {self.notch_freqs[0]} Hz notch filter applied.')
                    return notch_and_hp(raw_data, l_freq=None, h_freq=None, notch_freqs=self.notch_freqs,
                                        notch_widths=self.notch_widths, filter_type=self.filter_type)
                else:
                    pipeline_log.info('--> No temporal filtering applied.')

            data_list = []
            for f in filenames:
                raw = load_file(f)

                # If filtering separately, apply filters to each raw dataset before concatenation
                if self.filter_raws_separately:
                    filtered_raw = apply_filtering(raw)
                    data_list.append(filtered_raw)
                    if len(data_list) > 1:
                        data = mne.concatenate_raws(data_list, verbose='warning')
                    else:
                        data = filtered_raw
                else:
                    # No filtering yet, just append raw data to list and concatenate
                    data_list.append(raw)
                    if len(data_list) > 1:
                        data = mne.concatenate_raws(data_list, verbose='warning')
                        data = apply_filtering(data)

            # Save the filtered data to raw MNE-Python file
            data.save(pjoin(self.results_savepath, f'{data_type}_filtered.fif'), overwrite=True)

            # Plot filtered data artifacts
            save_eog_plot(pjoin(self.results_savepath, f'{data_type}_eog_filtered.png'), data)

            del data_list, raw; gc.collect()

            return data

        self.data = {}
        # Decide which data to load using values from parameter file
        # Iterate over filenames in list, concatenate separate recordings of same type, and apply filtering
        if self.data_types['resting_state']:
            if self.data_types['pre-treatment']:
                filenames_pre = [f for f in os.listdir(self.input_path) if 'pre' in f.split('_')]
                self.data['pre-treatment'] = filter_and_concatenate(filenames_pre, 'pre-treatment')
                pipeline_log.info(f'Pre-treatment resting state data shape: {self.data['pre-treatment']._data.shape}')
                save_psd(pjoin(self.results_savepath, 'psd_filtered_restingstate_pre.png'), self.data['pre-treatment'])
            if self.data_types['post-treatment']:
                filenames_post = [f for f in os.listdir(self.input_path) if 'post' in f.split('_')]
                self.data['post-treatment'] = filter_and_concatenate(filenames_post, 'post-treatment')
                pipeline_log.info(f'Post-treatment resting state data shape: {self.data['post-treatment']._data.shape}')
                save_psd(pjoin(self.results_savepath, 'psd_filtered_restingstate_post.png'), self.data['post-treatment'])
            elif self.data_from_TMS or self.data_from_motor or self.data_from_fif:
                filenames_other = [f for f in os.listdir(self.input_path) if 'treatment' in f]
                self.data['other'] = filter_and_concatenate(filenames_other, 'other')
                pipeline_log.info(f'Data shape: {self.data['other']._data.shape}')
                save_psd(pjoin(self.results_savepath, 'psd_filtered.png'), self.data['other'])
            else:
                raise ValueError('Data type is misspecified or does not exist in the input folder.')

    def run(self):
        # Clear screen and print basic data paths and parameter file location
        co.clear_screen()
        pipeline_log.info((co.color('purple', co.bold_text('CLEAN preprocessing pipeline.'))))
        pipeline_log.info(co.color('green', 'Paths'))
        pipeline_log.info((co.color('white', 'Configuration file: ')) + self.parameters_file)
        pipeline_log.info((co.color('white', 'Output Directory: ')) + self.results_savepath)
        pipeline_log.info('')

        # Run through pipline steps, printing successful stages and settings to console and log
        # Load data files provided as lists of paths to MFF files
        pipeline_log.info(co.color('periwinkle', 'Loading and filtering specified data from input directory...'))
        self.load_and_filter_data()

        # Iterate over self.data dictionary containing loaded raw data
        # With each key corresponding to pre-treatment resting state, post-treatment resting state, or other recording types
        for key in self.data:
            data = self.data[key]

            # Resample data
            data = self.resample_data(data, key)

            # Re-reference data to average reference
            pipeline_log.info(co.color('periwinkle', f'Re-referencing {key} data to average reference...'))
            self._reference(data, key, reference_method='average')

            # Finding bad channels using NoisyChannels
            if self.screen_bad_channels:
                self._find_bad_channels(data, key)
                pipeline_log.info((co.color('white', 'Done.')))
            else:
                pass

            # Segment data into epochs, identify remaining bad segments, and interpolate
            if self.bad_segment_interpolation:
                pipeline_log.info(co.color('periwinkle',
                                           f'Identifying and interpolating bad segments in {key} data...'))
                data = self._interpolate_bad_segments(data, key)
                pipeline_log.info((co.color('white',
                                            f'Finished bad segment identification and interpolation for {key} data.')))
            else:
                pipeline_log.info((co.color('periwinkle',
                                            f'Not running bad segment identification and interpolation on {key} data.')))

            # Run ICALabel to detect bad ICs
            if self.icalabel:
                pipeline_log.info(co.color('periwinkle', f'Running ICALabel cleaning on {key} data (this may take some time)...'))
                data = self._iclabel(data, key)
                pipeline_log.info((co.color('white', f'Finished ICALabel for {key} data.')))
            else:
                pipeline_log.info(co.color('periwinkle', 'Not running ICA...'))
            # Add : plot histogram of classification accuracies
            pipeline_log.info('')

            # Run wavelet ICA
            if self.wICA:
                pipeline_log.info(co.color('periwinkle', f'Running wavelet ICA cleaning with {self.wICA_num_components} components (this may take some time)...'))
                data = self._wICA(data, key)
                pipeline_log.info((co.color('white', 'Finished wICA.')))
            else:
                pipeline_log.info(co.color('periwinkle', 'Not running wICA...'))

            # Run ASR
            if self.asr:
                pipeline_log.info(co.color('periwinkle', f'Running artifact subspace reconstruction with cutoff {self.asr_cutoff}.'))
                data = self._artifact_subspace_reconstruction(data, key)
                pipeline_log.info((co.color('white', 'Finished ASR.')))
            else:
                pipeline_log.info(co.color('periwinkle', 'Not running artifact subspace reconstruction...'))

            # Overwrite filtered raw data stored in the corresponding self.data key with clean data
            self.data[key] = data

            # Preprocessing of specified data complete
            pipeline_log.info(co.color('periwinkle', f'Preprocessing of {key} data complete!'))

    # Resample if necessary
    def resample_data(self, data, name):
        if self.resample:
            pipeline_log.info(f'Resampling {name} data to {self.resample_rate} Hz.')
            data.resample(self.resample_rate)
        return data

    def _find_bad_channels(self, data, name):
        # Estimate bad channels
        pipeline_log.info(co.color('periwinkle', f'Using NoisyChannels to identify bad channels in {name}.'))
        nc = NoisyChannels(data, random_state=42)
        if not self.memory_intensive:
            nc.find_all_bads(channel_wise=True)
        else:
            nc.find_all_bads(channel_wise=False)

        self.bad_channels = nc.bad_by_deviation + nc.bad_by_hf_noise + nc.bad_by_dropout + nc.bad_by_ransac + nc.bad_by_correlation
        #nc.find_bad_by_correlation(frac_bad=0.05)
        #self.bad_channels += nc.bad_by_correlation

        # Print and log bad channel info
        pipeline_log.info((co.color('white', 'Removed ' + str(len(self.bad_channels)) + ' total:')))
        pipeline_log.info((co.color('white', '--> Removed ' + str(len(nc.bad_by_deviation)) + ' channels found to be bad by deviation.')))
        pipeline_log.info((co.color('white', '--> Removed ' + str(len(nc.bad_by_hf_noise)) + ' channels found to be bad by high frequency noise.')))
        pipeline_log.info((co.color('white', '--> Removed ' + str(len(nc.bad_by_correlation)) + ' channels found to be bad by correlations.')))
        pipeline_log.info((co.color('white', '--> Removed ' + str(len(nc.bad_by_dropout)) + ' channels found to be bad by dropout.')))
        pipeline_log.info((co.color('white', '--> Removed ' + str(len(nc.bad_by_ransac)) + ' channels found to be bad by RANSAC.')))

        # Interpolate bad channels
        data.info['bads'].extend(self.bad_channels)
        print(f'Bad channels in {name}: {self.bad_channels}')
        data.interpolate_bads(reset_bads=True)  # This will clear out data.info['bads']
        data.info['bads'] = self.bad_channels_list  # Reset data.info['bads'] to always bad channels
        print("Resting state pre-treatment data shape after interpolation:", data._data.shape)
        adj_num = data._data.shape[0] - len(self.bad_channels)

        # Adjust number of independent components
        if self.wICA_num_components > adj_num:
            pipeline_log.info((co.color('white', '  Adjusted number of wICA components to '+str(adj_num)+'.')))
            self.wICA_num_components = adj_num
        if self.iclabel_num_components > adj_num:
            pipeline_log.info((co.color('white', '  Adjusted number of ICALabel components to '+str(adj_num)+'.')))
            self.iclabel_num_components = adj_num

    def _wICA(self, data, name):
        pipeline_log.info((co.color('white', f'--> Fitting ICA for wICA on {name}...')))

        ica = FastICA(n_components=self.wICA_num_components, whiten="arbitrary-variance")
        ica.fit(data.get_data().T)
        ICs = ica.fit_transform(data.get_data().T)

        pipeline_log.info((co.color('white', '--> Removing wICA estimated artifacts...')))
        wICs, artifacts = wICA(ica, ICs)
        data._data -= artifacts.T

        pipeline_log.info((co.color('white', '--> Generating wICA plots...')))

        # Save sparkline plots of the IC timeseries pre_ICA, and the resulting thresholded IC timeseries.
        save_sparkline_ica(pjoin(self.results_savepath, f'{name}_ICA_timeseries_pre_wICA.png'), ICs)
        save_sparkline_ica(pjoin(self.results_savepath, f'{name}_wICA_artifact_timeseries.png'), wICs.T)

        # Save the wavelet-ICA-cleaned raw MNE-Python file
        pipeline_log.info((co.color('white', '--> Saving wICA cleaned FIF file...')))
        data.save(pjoin(self.results_savepath, f'{name}_wICA_cleaned.fif'), overwrite=True)

        # Generate artifact plots for wICA cleaned data
        pipeline_log.info((co.color('white', '--> Saving wICA cleaned FIF file...')))
        save_eog_plot(pjoin(self.results_savepath, f'{name}_eog_wICA_cleaned.png'), data)
        save_sparkline_ica(pjoin(self.results_savepath, f'{name}_wICA_cleaned_eeg_timeseries.png'), data._data)

        pipeline_log.info((co.color('white', f'--> Finished wICA on {name}.')))

    def _iclabel(self, data, name):
        # Use copy of data to avoid modifying in place
        data_copy = data.copy()

        # Generate artifact plots for data before cleaning
        save_eog_plot(pjoin(self.results_savepath, f'{name}_eog_icalabel_precleaning.png'), data)

        # Run ICALabel on the current data
        self.ic_labels, ic_ica, ic_cleaned = iclabel(data_copy, num_components=self.iclabel_num_components)

        # Save ICA topoplots and a folder of individual IC statistic plots
        save_ica_components(pjoin(self.results_savepath, f'{name}_ICALabel_ICA_topoplots.png'), ic_ica,
                            ic_label_obj=self.ic_labels)
        ic_save_path = pjoin(self.results_savepath, f'{name}_ICALabel_ICA_topoplots/')
        if not os.path.exists(ic_save_path):
            os.makedirs(ic_save_path)

        for i, label in enumerate(self.ic_labels['labels']):
            pipeline_log.info(f"Processing IC {i}")
            save_single_ic_plot(pjoin(ic_save_path, 'IC' + str(i).zfill(3) + '.png'), ic_ica,
                                data_copy, component_number=i, ic_label_obj=self.ic_labels)
            plt.close()

        # Generate artifact plots for cleaned data
        save_eog_plot(pjoin(self.results_savepath, f'{name}_eog_icalabel_cleaned.png'), ic_cleaned)

        # Generate histogram of icalabel prediction probabilities
        save_icalabel_prob_hist(pjoin(self.results_savepath, f'{name}_icalabel_probability_histogram.png'), self.ic_labels)

        # Save the ICALabel-cleaned raw MNE-Python file
        ic_cleaned.save(pjoin(self.results_savepath, f'{name}_icalabel_cleaned.fif'), overwrite=False)

        # Clean up
        del ic_ica; gc.collect()

        return ic_cleaned

    def _reference(self, data, name, reference_method):
        data.set_eeg_reference(reference_method)

    def _interpolate_bad_segments(self, data, name):
        epochs, ar = autoreject_bad_segments(data, method=self.segment_interpolation_method)
        epochs.save(pjoin(self.results_savepath, f'{name}_epochs_cleaned_interpolated.fif'), overwrite=True)

        # Construct new RAW object from epoched data output by autoreject
        X = np.concatenate(epochs.get_data(), axis=1)  # new numpy array of appended epochs

        self.montage = data.get_montage()
        data = mne.io.RawArray(X, data.info)
        data.set_montage(self.montage)

        if self.memory_intensive:
            data.save(pjoin(self.results_savepath, f'{name}_data_post_autoreject_raw.fif'), overwrite=True)

        # Generate rejected segment plot
        save_autoreject_plot(pjoin(self.results_savepath, f'{name}_autoreject_segments.png'), epochs, ar)
        # Generate artifact plots for cleaned data and data without cleaning
        save_eog_plot(pjoin(self.results_savepath, f'{name}_eog_autoreject_segments_cleaned.png'), data)

        del X; del epochs; gc.collect()

        return data


    def _artifact_subspace_reconstruction(self, data, name):
        # Run ASR
        data_cleaned = artifact_subspace_reconstruction(data, cutoff=self.asr_cutoff)

        # Generate artifact plots for cleaned data and data without cleaning
        save_eog_plot(pjoin(self.results_savepath, 'pre_ASR.png'), data)
        save_eog_plot(pjoin(self.results_savepath, 'post_ASR.png'), data_cleaned)

        # Replace pipeline data with new cleaned data
        data = data_cleaned
        del data_cleaned; gc.collect()

        # Save the ASR-cleaned raw MNE-Python file
        data.save(pjoin(self.results_savepath, f'{name}_ASR_cleaned.fif'), overwrite=True)


# ------------- Filtering functions ------------- #
def notch_and_hp(raw, notch_freqs, notch_widths, l_freq, h_freq, filter_type='fir'):
    notch_freqs = np.array(notch_freqs)
    notch_widths = np.array(notch_widths)
    raw_notch = raw.copy().notch_filter(freqs=notch_freqs, notch_widths=notch_widths, verbose='warning')
    raw_hp = raw_notch.filter(l_freq=l_freq, h_freq=h_freq, method=filter_type, verbose='warning')
    return raw_hp


# ------------- Wavelet ICA functions ------------- #
def ddencmp(x, wavelet='db1', scale=1.0):
    '''
    Python recreation of MATLAB's ddencmp function for choosing wavelet threshold using 
    Donoho and Johnstone universal threshold scaled by a robust variance estimate.
    Arg 'scale' allows adjusting the threshold more manually.

    '''
    n = len(x)
    (cA, cD)  = pywt.dwt(x,wavelet)
    noiselev = np.median(np.abs(cD))/0.6745
    thresh = np.sqrt(2*np.log(len(x)))*noiselev*scale
    return thresh


def wICA(ica, ICs, levels=5, wavelet='coif5', normalize=False, 
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
        print('  Data padded with ', str(len(extra)), ' additional values.')
        print('  Wavelet thresholding with wavelet:', wavelet)
    wavelet = pywt.Wavelet(wavelet)

    # Iterate over independent components, thresholding each one using a stationary wavelet transform
    # with soft thresholding.
    print('  Fitting ICA for wavelet-ICA cleaning...')
    wICs = []
    for i in range(ICs.shape[1]):
        if verbose:
            print("  Thresholding IC#",i)
        sig = np.concatenate((ICs[:,i],extra))
        thresh = ddencmp(sig, wavelet)
        swc = pywt.swt(sig, wavelet, level=levels, start_level=0, trim_approx=trim_approx, norm=normalize)
        y = pywt.threshold(swc, thresh, mode=thresholding, substitute=0)
        wICs.append(pywt.iswt(y, wavelet, norm=normalize))
    wICs = np.asarray(wICs)[:,0:ICs.shape[0]].T

    print("  Computing cleaned inverse transform...")
    artifacts = ica.inverse_transform(ICs)

    #artifacts = np.dot(ICs, ica.mixing_.T)

    #mean_tol = 1e-6
    #if ica.whiten:
    #    for i, mean in enumerate(ica.mean_):
    #        print("  adding back mean ", str(i))
    #        if np.abs(mean) > mean_tol:
    #            artifacts[:,i] = artifacts[:,i] + mean

    return wICs, artifacts

# ------------- ICA Label functions ------------- #
def iclabel(mne_raw, num_components=20, keep=["brain"], method='infomax', fit_params=dict(extended=True), reject=None):
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
    # Band pass data to band consistent with mne-icalabel training data
    mne_filt = mne_raw.filter(l_freq=1.0, h_freq=100.0, verbose='warning')

    # Fit ICA to mne_raw
    print('  Fitting ICA with ' + str(num_components) + ' components.')
    ica = ICA(n_components=num_components, max_iter="auto", method=method, random_state=42, fit_params=fit_params)
    ica.fit(mne_filt, reject=reject)

    # Use label_components function from ICLabel library to classify ICs and get their estimated probabilities
    print('  Using ICALabel to classify independent components.')
    ic_labels = label_components(mne_filt, ica, method="iclabel")
    labels = ic_labels["labels"]

    # Exclude ICs not in 'keep' list and reconstruct cleaned raw data
    exclude_idx = [idx for idx, label in enumerate(labels) if label not in keep]
    print(f"  Excluding these ICA components: {exclude_idx}")
    print(f'Number of ICA components excluded: {len(exclude_idx)}')
    cleaned = ica.apply(mne_raw, exclude=exclude_idx, verbose=False)
    return ic_labels, ica, cleaned

def autoreject_bad_segments(raw, segment_length=1.0, method='autoreject'):
    # generate epochs object from raw
    epochs = mne.make_fixed_length_epochs(raw, duration=segment_length, preload=True)

    # Use mne-compatible autoreject library to clean epoched data
    n_interpolates = np.array([1, 4, 32])
    consensus_percs = np.linspace(0, 1.0, 11)

    if method == 'autoreject':
        ar = AutoReject(n_interpolates, consensus_percs, thresh_method='bayesian_optimization', random_state=42)
    elif method == 'ransac':
        ar = Ransac()
    else:
        raise pipeline_log.ValueError("Specified bad segment method not implemented. Current options are 'autoreject' and 'ransac'.")
    epochs = ar.fit_transform(epochs)

    del raw; gc.collect()

    return epochs, ar

#def artifact_subspace_reconstruction(raw, cutoff=20):
#    '''
#    Apply artifact subspace reconstrution method using a Python implementation of EEGLAB's clear_rawdata ASR method. 
#    See: https://github.com/DiGyt/asrpy.  
#    '''
#    import asrpy
#    asr = asrpy.ASR(sfreq=raw.info["sfreq"], cutoff=cutoff)
#    asr.fit(raw)   
#    raw = asr.transform(raw)
#    return raw

#------------- Plotting functions -------------#
def save_psd(save_file, mne_raw, xlim=[0,300]):
    '''
    Save a power spectral density plot of the time series in mne_raw to the specified absolute file path
    using MNE-Python's compute_psd function.

    '''
    fig = mne_raw.compute_psd().plot(show=False)
    axs = fig.get_axes()
    axs[0].set_xlim(xlim)
    sns.despine()
    plt.savefig(save_file)
    plt.close()

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
    plt.close()

def save_ica_components(save_file, ica, ic_label_obj=None):
    '''
    Save a set of independent component plots using MNE-Python's plot_components method for ICA objects.
    Args:
        save_file: absolute path to save PNG output file.
        ica: an MNE-Python ica object.
        ic_label_list: optional list of conformal labels from ica_label to add to plot topomap titles.
    
    '''
    num_components = ica.n_components
    if ic_label_obj is not None:
        ic_label_list = ic_label_obj['labels']
        ic_label_probs = np.round(100*ic_label_obj['y_pred_proba'])
    # Iterate over batches of 20 ICs, making a plot for each group and adding labels and label probabilities from ICALabel
    for i in range(int(num_components/20)+1):
        if (i+1)*20 < num_components:
            picks = list(range(i*20,(i+1)*20))
            if len(picks) > 0:  
                fig = ica.plot_components(picks=picks, show=False)
        else:
            picks = list(range(i*20,num_components))
            if len(picks) > 0:
                fig = ica.plot_components(picks=picks, show=False)
        if ic_label_obj is not None:
            if len(picks) > 0:
                ic_label_list_part = ic_label_list[(i*20):((i+1)*20)]
                ic_probs_list_part = ic_label_probs[(i*20):((i+1)*20)]
                axs = fig.get_axes()
                for j,ax in enumerate(axs):
                    current_title = ax.get_title()
                    ax.set_title(current_title+':\n '+ic_label_list_part[j]+' ('+str(ic_probs_list_part[j])+'%)')
            plt.savefig(save_file.split('.')[0]+'_'+str(i).zfill(2)+save_file.split('.')[1])
            plt.close()

def save_single_ic_plot(save_file, ica, mne_raw, component_number=0, ic_label_obj=None):
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
    if ic_label_obj is not None:
        ic_label = ic_label_obj['labels'][component_number]
        ic_prob = ic_label_obj['y_pred_proba'][component_number]
        axs = fig[0].get_axes()
        current_title = axs[0].get_title()
        axs[0].set_title(current_title+': '+ic_label+' ('+str(ic_prob)+'%)')
    plt.savefig(save_file)
    plt.close()

def save_eog_plot(save_file, mne_in, channel_list=['E32', 'E241', 'E25', 'E238']):
    '''
    Save an eog artifact plot for the time series in mne_raw to the specified absolute file path.
    Uses MNE-Python's create_eog_epochs and plot_joint functions and defaults to channels 
    ['E32','E241','E25','E238'] which correspond to EGI's 256 channel hydrocel net eye motion channels.

    '''
    average_eog = mne.preprocessing.create_eog_epochs(mne_in,ch_name=channel_list).average()
    average_eog.plot_joint(show=False)
    sns.despine()
    plt.savefig(save_file)
    plt.close()

def save_icalabel_prob_hist(save_file, ic_label_obj):
    plt.figure(figsize=(4,4))
    ic_label_probs = ic_label_obj['y_pred_proba']
    # TODO: break this into probabilities for each class ICAlabel produces.
    plt.hist(ic_label_probs, bins=30)
    sns.despine()
    plt.savefig(save_file)
    plt.close()

def save_autoreject_plot(save_file, epochs, ar_obj):
    ar_obj.get_reject_log(epochs).plot(show=False)
    plt.savefig(save_file)
    plt.close()

#-------------------------------------------------------------------------------                                     
# Main                                                                                                               

if __name__ == '__main__':
    # Parse command line                                                                                             
    from argparse import ArgumentParser
    from argparse import SUPPRESS
    parser = ArgumentParser()

    pred_pipeline = Preprocessing('test_config.cfg')
    pred_pipeline.run()
