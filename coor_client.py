from coor import coor_reader, coor_plotter, coor_info

"""
For UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 3131: invalid start byte
on mac, delete hidden .DS_Store file from input directory
"""

# Get current absolute path and set path to input_files directory
data_folder = '/Users/Bella/Desktop/Grosenick_Lab/eeg_patient_data/coor_files/input_files'


# Get patient ID: file path dictionary
patient_data_filepaths = coor_reader.readbrainfolder(data_folder)

# Iterate through patient id list
for patient in patient_data_filepaths:
    samples_df = coor_reader.loadbrainfile(patient_data_filepaths, patient)

    # Plot average coil position changes per day
    coor_plotter.plot_coor_days(samples_df, patient, data_folder)

    # Plot coil position changes per session per day
    coor_plotter.plot_coor_sessions(samples_df, patient, data_folder)

    # Get sample statistics and write to new file
    coor_info.write_stats(samples_df, patient, data_folder)
