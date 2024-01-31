import os
import pandas as pd


# Reads data folder containing Brainsite files, one file per
# patient, and creates a dictionary mapping patient ID to file path
def readbrainfolder(data_folder):
    # List directory contents
    dir_contents = os.listdir(data_folder)

    patient_data_filepaths = {}
    for item in dir_contents:
        patient_id = item.split('_')[0]
        patient_data_filepaths[patient_id] = os.path.join(data_folder, item)

    return patient_data_filepaths


# Takes a BrainSite txt file for one patient
# and returns a pandas dataframe with relevant coordinate data
def loadbrainfile(patient_data_filepaths, patient_id):
    brainsite = patient_data_filepaths[patient_id]

    with open(brainsite) as brainsite:
        # Skip descriptor lines, strip newlines, separate entries by tab
        brainfile = brainsite.readlines()[6:]
        brainfile = [line.strip('\n') for line in brainfile]
        brainfile = [line.split('\t') for line in brainfile]

        # Define headers and separate relevant sample_info
        if patient_id != 'Bart':
            sample_info = brainfile[3:]
            sample_headers = brainfile[2]

        # Bart is missing two header rows
        elif patient_id == 'Bart':
            sample_headers = brainfile[0]
            sample_info = brainfile[1:]

        sample_floated = []

        # Iterate over sample_info to convert numeric strings to floats
        for row in sample_info:
            floated_row = []
            for entry in row:
                try:
                    floated_row.append(round(float(entry), 2))
                except ValueError:
                    floated_row.append(entry)
                    continue

            sample_floated.append(floated_row)

    samples_df = pd.DataFrame(sample_floated, columns=sample_headers)

    # Regex pattern to identify only valid session names
    valid_sessions_regex = r'^Session\s\d+.*'

    # Drop rows with NaN values
    # Keep only the rows that have a valid session name
    samples_df = samples_df.dropna()
    try:
        samples_df = samples_df.loc[samples_df['Session Name'].str.match(valid_sessions_regex)]
    except KeyError:
        print(f'Error retreiving sessions for {patient_id}')
        print(samples_df)

    # Keep only relevant columns
    samples_df.drop(['Creation Cause', 'Stim. Power B', 'Stim. Pulse Interval',
                     'Stim. Power A', 'Offset'], axis=1, inplace=True)

    return samples_df
