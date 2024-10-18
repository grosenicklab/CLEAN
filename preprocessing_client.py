import os
import re
import configparser
import subprocess

to_preprocess_datapath = '/Users/Bella/Desktop/Grosenick_Lab/eeg_patient_data/mdd_dlpfc/to_preprocess'


# Takes a directory file path holding subject data that needs to be preprocessed
# Returns a list of the subject file paths
def get_subject_paths(datapath):
    subject_paths = []
    dir_contents = os.listdir(datapath)
    for folder in dir_contents:
        if folder != '.DS_Store':
            folder_path = os.path.join(datapath, folder)
            subject_paths.append(folder_path)
    return subject_paths


# Takes a path to one subject's data and returns the filepaths for each day data directory
def get_day_paths(subject_path):
    regex = r'[cm]{1}\d{3}_[a-zA-Z]+_day\d{1}'
    dircontents = os.listdir(subject_path)
    day_paths = [folder for folder in dircontents if re.match(regex, folder)]
    day_paths.sort(key=lambda x: x.split('_')[-1])
    return day_paths


# Load and modify the config file
def update_config(preprocess_path):
    config_file = 'test_config.cfg'  # Specify the path to your config file
    config = configparser.ConfigParser()
    config.read(config_file)

    # Update the input_path and results_path in the config file
    config['paths']['input_path'] = preprocess_path
    config['paths']['results_path'] = os.path.join(preprocess_path, 'Results')

    # Write the updated config back to the file
    with open(config_file, 'w') as configfile:
        config.write(configfile)


# Run preprocessing.py with updated config
def run_preprocessing():
    try:
        subprocess.run(['python', 'preprocessing.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f'Error during preprocessing: {e}')


# Run preprocessing for subjects in to_preprocess
subject_paths = get_subject_paths(to_preprocess_datapath)

for subject_path in subject_paths:
    day_paths = get_day_paths(subject_path)
    for day_path in day_paths:
        preprocess_path = os.path.join(subject_path, day_path)
        print(f'Preprocessing subject {subject_path.split('/')[-1]}, {day_path.split('_')[-1]}')
        update_config(preprocess_path)
        run_preprocessing()
        print(f'Completed preprocessing for {subject_path.split('/')[-1]}, {day_path.split('_')[-1]}')
